//@ run-pass
// We disable tail merging here because it can't preserve debuginfo and thus
// potentially breaks the backtraces. Also, subtle changes can decide whether
// tail merging succeeds, so the test might work today but fail tomorrow due to a
// seemingly completely unrelated change.
// Unfortunately, LLVM has no "disable" option for this, so we have to set
// "enable" to 0 instead.

//@ compile-flags:-g -Copt-level=0 -Cllvm-args=-enable-tail-merge=0
//@ compile-flags:-Cforce-frame-pointers=yes
//@ compile-flags:-Cstrip=none
//@ ignore-android FIXME #17520
//@ needs-subprocess
//@ ignore-fuchsia Backtrace not symbolized, trace different line alignment
//@ ignore-ios needs the `.dSYM` files to be moved to the device
//@ ignore-tvos needs the `.dSYM` files to be moved to the device
//@ ignore-watchos needs the `.dSYM` files to be moved to the device
//@ ignore-visionos needs the `.dSYM` files to be moved to the device

// FIXME(#117097): backtrace (possibly unwinding mechanism) seems to be different on at least
// `i686-mingw` (32-bit windows-gnu)? cc #128911.
//@ ignore-windows-gnu
//@ ignore-backends: gcc
//@ ignore-msvc see #62897 and `backtrace-debuginfo.rs` test

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
