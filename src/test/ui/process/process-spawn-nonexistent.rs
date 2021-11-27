// run-pass
// ignore-emscripten no processes
// ignore-sgx no processes

use std::io::ErrorKind;
use std::process::Command;

fn main() {
    let err = Command::new("nonexistent")
        .spawn()
        .unwrap_err()
        .kind();

    assert!(err == ErrorKind::NotFound || err == ErrorKind::PermissionDenied);
}
