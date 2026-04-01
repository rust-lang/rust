use crate::os::unix::process::{CommandExt, ExitStatusExt};
use crate::panic::catch_unwind;
use crate::process::Command;

// Many of the other aspects of this situation, including heap alloc concurrency
// safety etc., are tested in tests/ui/process/process-panic-after-fork.rs

#[test]
fn exitstatus_display_tests() {
    // In practice this is the same on every Unix.
    // If some weird platform turns out to be different, and this test fails, use #[cfg].
    use crate::os::unix::process::ExitStatusExt;
    use crate::process::ExitStatus;

    let t = |v, s| assert_eq!(s, format!("{}", <ExitStatus as ExitStatusExt>::from_raw(v)));

    t(0x0000f, "signal: 15 (SIGTERM)");
    t(0x0008b, "signal: 11 (SIGSEGV) (core dumped)");
    t(0x00000, "exit status: 0");
    t(0x0ff00, "exit status: 255");

    // On MacOS, 0x0137f is WIFCONTINUED, not WIFSTOPPED. Probably *BSD is similar.
    //   https://github.com/rust-lang/rust/pull/82749#issuecomment-790525956
    // The purpose of this test is to test our string formatting, not our understanding of the wait
    // status magic numbers. So restrict these to Linux.
    if cfg!(target_os = "linux") {
        #[cfg(any(target_arch = "mips", target_arch = "mips64"))]
        t(0x0137f, "stopped (not terminated) by signal: 19 (SIGPWR)");

        #[cfg(any(target_arch = "sparc", target_arch = "sparc64"))]
        t(0x0137f, "stopped (not terminated) by signal: 19 (SIGCONT)");

        #[cfg(not(any(
            target_arch = "mips",
            target_arch = "mips64",
            target_arch = "sparc",
            target_arch = "sparc64"
        )))]
        t(0x0137f, "stopped (not terminated) by signal: 19 (SIGSTOP)");

        t(0x0ffff, "continued (WIFCONTINUED)");
    }

    // Testing "unrecognised wait status" is hard because the wait.h macros typically
    // assume that the value came from wait and isn't mad. With the glibc I have here
    // this works:
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        t(0x000ff, "unrecognised wait status: 255 0xff");
    }
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
#[cfg_attr(any(target_os = "tvos", target_os = "watchos"), ignore = "fork is prohibited")]
fn test_command_fork_no_unwind() {
    let got = catch_unwind(|| {
        let mut c = Command::new("echo");
        c.arg("hi");
        unsafe {
            c.pre_exec(|| panic!("{}", "crash now!"));
        }
        let st = c.status().expect("failed to get command status");
        dbg!(st);
        st
    });
    dbg!(&got);
    let status = got.expect("panic unexpectedly propagated");
    dbg!(status);
    let signal = status.signal().expect("expected child process to die of signal");
    assert!(
        signal == libc::SIGABRT
            || signal == libc::SIGILL
            || signal == libc::SIGTRAP
            || signal == libc::SIGSEGV
    );
}
