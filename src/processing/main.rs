use onnxruntime::{environment::Environment, ndarray::Array, GraphOptimizationLevel, LoggingLevel, session::Session};
use zmq::{Context, Socket};
use std::time::Duration;
use std::env;
use std::process::exit;
use nix::unistd::{fork, ForkResult, setsid};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Daemonize the process
    daemonize();

    // Run the resilient processing loop
    run_processing_loop()
}

fn daemonize() {
    // First fork
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => exit(0), // Exit the parent
        Ok(ForkResult::Child) => (),
        Err(_) => panic!("First fork failed"),
    }

    // Create a new session
    setsid().expect("Failed to create new session");

    // Second fork
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => exit(0), // Exit the second parent
        Ok(ForkResult::Child) => (),
        Err(_) => panic!("Second fork failed"),
    }

    // The process is now daemonized
}

fn run_processing_loop() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    let zmq_address = env::var("ZMQ_ADDRESS").unwrap_or("tcp://localhost:5555".to_string());
    let model_path = env::var("MODEL_PATH").unwrap_or("model.onnx".to_string());

    // Initialize ZeroMQ
    let context = Context::new();
    let socket = setup_socket(&context, &zmq_address)?;

    // Set up ONNX Runtime environment and load model with retries
    let environment = Environment::builder()
        .with_name("InferenceEnvironment")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .expect("Failed to create ONNX Runtime environment");

    let session = setup_model(&environment, &model_path)?;

    // Start processing loop
    loop {
        match socket.recv_bytes(0) {
            Ok(frame_data) => {
                match process_frame(&session, frame_data) {
                    Ok(outputs) => println!("{:?}", outputs), // Process outputs here or send results back via ZeroMQ
                    Err(e) => eprintln!("Error during inference: {:?}", e),
                }
            }
            Err(e) => {
                eprintln!("Failed to receive frame: {:?}", e);
                std::thread::sleep(Duration::from_secs(1)); // Retry after delay
            }
        }
    }
}

fn setup_socket(context: &Context, address: &str) -> Result<Socket, zmq::Error> {
    loop {
        match context.socket(zmq::PULL) {
            Ok(socket) => match socket.connect(address) {
                Ok(_) => return Ok(socket),
                Err(e) => {
                    eprintln!("Failed to connect to socket: {:?}. Retrying...", e);
                    std::thread::sleep(Duration::from_secs(2));
                }
            },
            Err(e) => {
                eprintln!("Error creating ZeroMQ socket: {:?}. Retrying...", e);
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}

fn setup_model(environment: &Environment, model_path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    loop {
        match environment.new_session_builder()
            .and_then(|builder| builder.with_graph_optimization_level(GraphOptimizationLevel::Basic))
            .and_then(|builder| builder.with_model_from_file(model_path)) {
            Ok(session) => return Ok(session),
            Err(e) => {
                eprintln!("Failed to load model from {}: {:?}. Retrying...", model_path, e);
                std::thread::sleep(Duration::from_secs(5)); // Retry after delay
            }
        }
    }
}

fn process_frame(session: &Session, frame_data: Vec<u8>) -> Result<Vec<Array<f32, _>>, Box<dyn std::error::Error>> {
    // Convert the frame into an input tensor
    let input_tensor = Array::from_shape_vec((1, 3, 224, 224), frame_data)?; // Adjust shape as needed

    // Run inference
    let outputs: Vec<Array<f32, _>> = session.run(vec![input_tensor.into_dyn()])?;
    Ok(outputs)
}
