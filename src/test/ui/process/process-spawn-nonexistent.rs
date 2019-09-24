// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::io::ErrorKind;
use std::process::Command;

fn main() {
    assert_eq!(Command::new("nonexistent")
                   .spawn()
                   .unwrap_err()
                   .kind(),
               ErrorKind::NotFound);
}
