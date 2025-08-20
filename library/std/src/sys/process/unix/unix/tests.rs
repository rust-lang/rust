use crate::fs;
use crate::os::unix::fs::MetadataExt;
use crate::os::unix::process::{CommandExt, ExitStatusExt};
use crate::panic::catch_unwind;
use crate::process::Command;
use crate::sys::AsInner;

// Many of the other aspects of this situation, including heap alloc concurrency
// safety etc., are tested in tests/ui/process/process-panic-after-fork.rs

/// Use dev + ino to uniquely identify a file
fn md_file_id(md: &fs::Metadata) -> (u64, u64) {
    (md.dev(), md.ino())
}

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

/// For `Command`'s fd-related tests, we want to be sure they work both with exec
/// and with `posix_spawn`. We test both the default which should use `posix_spawn`
/// on supported platforms, and using `pre_exec` to force spawn using `exec`.
mod fd_impls {
    use super::{assert_spawn_method, md_file_id};
    use crate::fs;
    use crate::io::{self, Write};
    use crate::os::fd::AsRawFd;
    use crate::os::unix::process::CommandExt;
    use crate::process::{Command, Stdio};

    /// Check setting the child's stdin via `.fd`.
    pub fn test_stdin(use_exec: bool) {
        let (pipe_reader, mut pipe_writer) = io::pipe().unwrap();

        let fd_num = libc::STDIN_FILENO;

        let mut cmd = Command::new("cat");
        cmd.stdout(Stdio::piped()).fd(fd_num, pipe_reader);

        if use_exec {
            unsafe {
                cmd.pre_exec(|| Ok(()));
            }
        }

        let mut child = cmd.spawn().unwrap();
        let mut stdout = child.stdout.take().unwrap();

        assert_spawn_method(&cmd, use_exec);

        pipe_writer.write_all(b"Hello, world!").unwrap();
        drop(pipe_writer);

        child.wait().unwrap().exit_ok().unwrap();
        assert_eq!(io::read_to_string(&mut stdout).unwrap(), "Hello, world!");
    }

    // FIXME: fails on android
    #[cfg_attr(not(target_os = "android"), should_panic)]
    /// Check that the last `.fd` mapping is preserved when there are conflicts.
    pub fn test_swap(use_exec: bool) {
        let (pipe_reader1, mut pipe_writer1) = io::pipe().unwrap();
        let (pipe_reader2, mut pipe_writer2) = io::pipe().unwrap();

        let num1 = pipe_reader1.as_raw_fd();
        let num2 = pipe_reader2.as_raw_fd();

        let mut cmd = Command::new("cat");
        cmd.arg(format!("/dev/fd/{num1}"))
            .arg(format!("/dev/fd/{num2}"))
            .stdout(Stdio::piped())
            .fd(num2, pipe_reader1)
            .fd(num1, pipe_reader2);

        if use_exec {
            unsafe {
                cmd.pre_exec(|| Ok(()));
            }
        }

        pipe_writer1.write_all(b"Hello from pipe 1!").unwrap();
        drop(pipe_writer1);

        pipe_writer2.write_all(b"Hello from pipe 2!").unwrap();
        drop(pipe_writer2);

        let mut child = cmd.spawn().unwrap();
        let mut stdout = child.stdout.take().unwrap();

        assert_spawn_method(&cmd, use_exec);

        child.wait().unwrap().exit_ok().unwrap();
        // the second pipe's output is clobbered; this is expected.
        assert_eq!(io::read_to_string(&mut stdout).unwrap(), "Hello from pipe 1!");
    }

    // ensure that the fd is properly closed in the parent, but only after the child is spawned.
    pub fn test_close_time(use_exec: bool) {
        let (_pipe_reader, pipe_writer) = io::pipe().unwrap();

        let fd = pipe_writer.as_raw_fd();
        let fd_path = format!("/dev/fd/{fd}");

        let mut cmd = Command::new("true");
        cmd.fd(123, pipe_writer);

        if use_exec {
            unsafe {
                cmd.pre_exec(|| Ok(()));
            }
        }

        // Get the identifier of the fd (metadata follows symlinks)
        let fd_id = md_file_id(&fs::metadata(&fd_path).expect("fd should be open"));

        cmd.spawn().unwrap().wait().unwrap().exit_ok().unwrap();

        assert_spawn_method(&cmd, use_exec);

        // After the child is spawned, our fd should be closed
        match fs::metadata(&fd_path) {
            // Ok; fd exists but points to a different file
            Ok(md) => assert_ne!(md_file_id(&md), fd_id),
            // Ok; fd does not exist
            Err(_) => (),
        }
    }
}

#[test]
fn fd_test_stdin() {
    fd_impls::test_stdin(false);
    fd_impls::test_stdin(true);
}

#[test]
fn fd_test_swap() {
    fd_impls::test_swap(false);
    fd_impls::test_swap(true);
}

#[test]
fn fd_test_close_time() {
    fd_impls::test_close_time(false);
    fd_impls::test_close_time(true);
}

#[track_caller]
fn assert_spawn_method(cmd: &Command, use_exec: bool) {
    let used_posix_spawn = cmd.as_inner().get_last_spawn_was_posix_spawn().unwrap();
    if use_exec {
        assert!(!used_posix_spawn, "posix_spawn used but exec was expected");
    } else if cfg!(any(
        target_os = "freebsd",
        target_os = "illumos",
        all(target_os = "linux", target_env = "gnu"),
        all(target_os = "linux", target_env = "musl"),
        target_os = "nto",
        target_vendor = "apple",
        target_os = "cygwin",
    )) {
        assert!(used_posix_spawn, "platform supports posix_spawn but it wasn't used");
    } else {
        assert!(!used_posix_spawn);
    }
}
