use opencv::prelude::*;
use opencv::videoio;
use opencv::core::Mat;
use zmq::{Context, Socket};
use std::time::Duration;
use std::process::exit;
use nix::unistd::{fork, ForkResult, setsid};

fn main() -> opencv::Result<()> {
    // Daemonize the process
    daemonize();

    // Run the resilient capture and send loop
    run_capture_loop()
}

fn daemonize() {
    // First fork
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => exit(0), // Exit the parent
        Ok(ForkResult::Child) => (),
        Err(_) => panic!("Fork failed"),
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

fn run_capture_loop() -> opencv::Result<()> {
    // Initialize ZeroMQ Context and Socket
    let context = Context::new();
    let socket = setup_socket(&context, "tcp://*:5555");

    // Initialize the webcam capture
    let mut cam = setup_camera();

    loop {
        // Attempt to capture a frame
        let mut frame = Mat::default();
        match cam.read(&mut frame) {
            Ok(_) if !frame.empty() => {
                // Encode the frame as bytes (e.g., as a JPEG) before sending
                let mut encoded = opencv::core::Vector::<u8>::new();
                if let Ok(_) = opencv::imgcodecs::imencode(".jpg", &frame, &mut encoded, &Default::default()) {
                    // Attempt to send the frame
                    if let Err(_) = try_send(&socket, &encoded) {
                        eprintln!("Failed to send frame, will retry later");
                    }
                }
            }
            Ok(_) => eprintln!("Empty frame captured, retrying..."),
            Err(_) => {
                eprintln!("Camera error, attempting to reconnect...");
                cam = setup_camera(); // Try to reinitialize camera if it fails
            }
        }

        // Approximate delay to maintain 30 FPS
        std::thread::sleep(Duration::from_millis(33));
    }
}

// Utility function to set up ZeroMQ socket with retries
fn setup_socket(context: &Context, address: &str) -> Socket {
    loop {
        match context.socket(zmq::PUSH) {
            Ok(socket) => {
                if socket.bind(address).is_ok() {
                    return socket;
                }
            }
            Err(_) => eprintln!("Failed to create or bind ZeroMQ socket, retrying..."),
        }
        std::thread::sleep(Duration::from_secs(1)); // Retry every second
    }
}

// Utility function to handle non-blocking send with retry
fn try_send(socket: &Socket, data: &opencv::core::Vector<u8>) -> Result<(), ()> {
    match socket.send(data.to_bytes(), zmq::DONTWAIT) {
        Ok(_) => Ok(()),
        Err(e) if e == zmq::Error::EAGAIN => {
            eprintln!("Socket is busy, will retry later");
            Err(())
        }
        Err(_) => {
            eprintln!("Error sending data, will retry later");
            Err(())
        }
    }
}

// Utility function to set up and initialize the camera with retries
fn setup_camera() -> videoio::VideoCapture {
    loop {
        match videoio::VideoCapture::new(0, videoio::CAP_ANY) {
            Ok(mut cam) => {
                cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0).ok();
                cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0).ok();
                return cam;
            }
            Err(_) => eprintln!("Failed to initialize camera, retrying..."),
        }
        std::thread::sleep(Duration::from_secs(1)); // Retry every second
    }
}
