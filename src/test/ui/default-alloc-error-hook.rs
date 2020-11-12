// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::alloc::{Layout, handle_alloc_error};
use std::env;
use std::process::Command;
use std::str;

fn main() {
    if env::args().len() > 1 {
        handle_alloc_error(Layout::new::<[u8; 42]>())
    }

    let me = env::current_exe().unwrap();
    let output = Command::new(&me).arg("next").output().unwrap();
    assert!(!output.status.success(), "{:?} is a success", output.status);
    assert_eq!(str::from_utf8(&output.stderr).unwrap(), "memory allocation of 42 bytes failed\n");
}
