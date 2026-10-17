use crate::os::unix::process::{CommandExt, ExitStatusExt};
use crate::panic::catch_unwind;
use crate::process::Command;

// Many of the other aspects of this situation, including heap alloc concurrency
// safety etc., are tested in tests/ui/process/process-panic-after-fork.rs

#[test]
fn exitstatus_debug_display_tests() {
    // In practice this is the same on every Unix.
    // If some weird platform turns out to be different, and this test fails, use #[cfg].
    use crate::os::unix::process::ExitStatusExt;
    use crate::process::ExitStatus;

    let t = |v, disp, dbg| {
        let status = <ExitStatus as ExitStatusExt>::from_raw(v);
        assert_eq!(disp, format!("{status}"), "from value: {v:#x}");
        assert_eq!(dbg, format!("{status:?}"), "from value: {v:#x}");
    };

    t(
        0x0000f,
        "signal: 15 (SIGTERM)",
        "ExitStatus(unix_wait_status { \
            value: 15, \
            status: Signaled { signal: 15, name: SIGTERM, core_dumped: false } \
        })",
    );
    t(
        0x0008b,
        "signal: 11 (SIGSEGV) (core dumped)",
        "ExitStatus(unix_wait_status { \
            value: 139, \
            status: Signaled { signal: 11, name: SIGSEGV, core_dumped: true } \
        })",
    );
    t(0x00000, "exit status: 0", "ExitStatus(unix_wait_status { value: 0, status: Exited(0) })");
    t(
        0x0ff00,
        "exit status: 255",
        "ExitStatus(unix_wait_status { value: 65280, status: Exited(255) })",
    );

    // On MacOS, 0x0137f is WIFCONTINUED, not WIFSTOPPED. Probably *BSD is similar.
    //   https://github.com/rust-lang/rust/pull/82749#issuecomment-790525956
    // The purpose of this test is to test our string formatting, not our understanding of the wait
    // status magic numbers. So restrict these to Linux.
    if cfg!(target_os = "linux") {
        if cfg!(any(target_arch = "mips", target_arch = "mips64")) {
            t(
                0x0137f,
                "stopped (not terminated) by signal: 19 (SIGPWR)",
                "ExitStatus(unix_wait_status { \
                    value: 4991, \
                    status: Stopped { signal: 19, name: SIGPWR } \
                })",
            );
        } else if cfg!(any(target_arch = "sparc", target_arch = "sparc64")) {
            t(
                0x0137f,
                "stopped (not terminated) by signal: 19 (SIGCONT)",
                "ExitStatus(unix_wait_status { \
                    value: 4991, \
                    status: Stopped { signal: 19, name: SIGCONT } \
                })",
            );
        } else {
            t(
                0x0137f,
                "stopped (not terminated) by signal: 19 (SIGSTOP)",
                "ExitStatus(unix_wait_status { \
                    value: 4991, \
                    status: Stopped { signal: 19, name: SIGSTOP } \
                })",
            );
        }

        t(
            0x0ffff,
            "continued (WIFCONTINUED)",
            "ExitStatus(unix_wait_status { \
                value: 65535, \
                status: Continued { name: WIFCONTINUED } \
            })",
        );
    }

    // Testing "unrecognised wait status" is hard because the wait.h macros typically
    // assume that the value came from wait and isn't mad. With the glibc I have here
    // this works:
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        t(
            0x000ff,
            "unrecognised wait status: 255 0xff",
            "ExitStatus(unix_wait_status { value: 255, status: Unrecognized(0xff) })",
        );
    }
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
#[cfg_attr(
    any(target_os = "tvos", target_os = "watchos", target_os = "l4re"),
    ignore = "fork is prohibited"
)]
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
