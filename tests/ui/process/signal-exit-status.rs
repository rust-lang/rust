//@ run-pass
//@ ignore-emscripten no processes
//@ ignore-sgx no processes
//@ ignore-windows
//@ ignore-fuchsia code returned as ZX_TASK_RETCODE_EXCEPTION_KILL, FIXME (#58590)

#![feature(core_intrinsics)]

use std::env;
use std::process::Command;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "signal" {
        // Raise an aborting signal without UB
        core::intrinsics::abort();
    } else {
        let status = Command::new(&args[0]).arg("signal").status().unwrap();
        assert!(status.code().is_none());
    }
}
