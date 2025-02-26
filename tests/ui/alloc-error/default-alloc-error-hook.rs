//@ run-pass
//@ needs-subprocess

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

    let mut stderr = str::from_utf8(&output.stderr).unwrap();

    // When running inside QEMU user-mode emulation, there will be an extra message printed by QEMU
    // in the stderr whenever a core dump happens. Remove it before the check.
    stderr = stderr
        .strip_suffix("qemu: uncaught target signal 6 (Aborted) - core dumped\n")
        .unwrap_or(stderr);

    assert_eq!(stderr, "memory allocation of 42 bytes failed\n");
}
