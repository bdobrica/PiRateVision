#!/bin/bash

sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libzmq3-dev libopencv-dev python3-opencv
sudo apt-get install -y onnxruntime

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
echo ". \"$HOME/.cargo/env\"" >> "$HOME/.bash_profile"
. "$HOME/.cargo/env"
rustup target add aarch64-unknown-linux-gnu

sudo apt install gcc-aarch64-linux-gnu
