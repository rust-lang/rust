//@ run-pass
//@ ignore-android FIXME #17520
//@ needs-subprocess
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-fuchsia Backtraces not symbolized
//@ compile-flags:-g
//@ compile-flags:-Cstrip=none

use std::alloc::{Layout, handle_alloc_error};
use std::process::Command;
use std::{env, str};

fn main() {
    if env::args().len() > 1 {
        handle_alloc_error(Layout::new::<[u8; 42]>())
    }

    let me = env::current_exe().unwrap();
    let output = Command::new(&me).env("RUST_BACKTRACE", "1").arg("next").output().unwrap();
    assert!(!output.status.success(), "{:?} is a success", output.status);

    let mut stderr = str::from_utf8(&output.stderr).unwrap();

    // When running inside QEMU user-mode emulation, there will be an extra message printed by QEMU
    // in the stderr whenever a core dump happens. Remove it before the check.
    stderr = stderr
        .strip_suffix("qemu: uncaught target signal 6 (Aborted) - core dumped\n")
        .unwrap_or(stderr);

    assert!(stderr.contains("memory allocation of 42 bytes failed"), "{}", stderr);
    assert!(stderr.contains("alloc_error_backtrace::main"), "{}", stderr);
}
