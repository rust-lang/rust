use std::os::unix::prelude::*;
use std::process::Command;
use std::env;

fn main() {
    let args = env::args()
        .skip(1)
        .filter(|s| s != "--strip-debug")
        .collect::<Vec<_>>();
    panic!("failed to exec: {}", Command::new("rust-lld").args(&args).exec());
}
