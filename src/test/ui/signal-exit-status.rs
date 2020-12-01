// run-pass
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-windows

use std::env;
use std::process::Command;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "signal" {
        // Raise a segfault.
        unsafe { *(1 as *mut isize) = 0; }
    } else {
        let status = Command::new(&args[0]).arg("signal").status().unwrap();
        assert!(status.code().is_none());
    }
}
