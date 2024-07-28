//@ run-pass
//@ ignore-wasm32 no processes
//@ ignore-sgx no processes
//@ ignore-windows

#![feature(core_intrinsics)]
#![cfg_attr(target_os = "fuchsia", feature(fuchsia_exit_status))]

use std::env;
use std::process::Command;

#[cfg(target_os = "fuchsia")]
use std::os::fuchsia::process::{ExitStatusExt, ZX_TASK_RETCODE_EXCEPTION_KILL};

pub fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "signal" {
        // Raise an aborting signal without UB
        core::intrinsics::abort();
    } else {
        // Spawn a child process that will raise an aborting signal
        let status = Command::new(&args[0]).arg("signal").status().unwrap();

        #[cfg(not(target_os = "fuchsia"))]
        assert!(status.code().is_none());

        // Upon abort(), a Fuchsia process will trigger a kernel exception
        // that, if uncaught, will cause the kernel to kill the process with
        // ZX_TASK_RETCODE_EXCEPTION_KILL. The same code could be
        // returned for a different unhandled exception, but the simplicity of
        // the program under test makes such an exception unlikely.
        #[cfg(target_os = "fuchsia")]
        assert_eq!(Some(ZX_TASK_RETCODE_EXCEPTION_KILL), status.task_retcode());
    }
}
