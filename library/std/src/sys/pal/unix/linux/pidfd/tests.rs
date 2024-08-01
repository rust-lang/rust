use crate::assert_matches::assert_matches;
use crate::os::fd::{AsRawFd, RawFd};
use crate::os::linux::process::{ChildExt, CommandExt as _};
use crate::os::unix::process::{CommandExt as _, ExitStatusExt};
use crate::process::Command;

#[test]
fn test_command_pidfd() {
    let pidfd_open_available = probe_pidfd_support();

    // always exercise creation attempts
    let mut child = Command::new("false").create_pidfd(true).spawn().unwrap();

    // but only check if we know that the kernel supports pidfds.
    // We don't assert the precise value, since the standard library
    // might have opened other file descriptors before our code runs.
    if pidfd_open_available {
        assert!(child.pidfd().is_ok());
    }
    if let Ok(pidfd) = child.pidfd() {
        let flags = super::cvt(unsafe { libc::fcntl(pidfd.as_raw_fd(), libc::F_GETFD) }).unwrap();
        assert!(flags & libc::FD_CLOEXEC != 0);
    }
    assert!(child.id() > 0 && child.id() < -1i32 as u32);
    let status = child.wait().expect("error waiting on pidfd");
    assert_eq!(status.code(), Some(1));

    let mut child = Command::new("sleep").arg("1000").create_pidfd(true).spawn().unwrap();
    assert_matches!(child.try_wait(), Ok(None));
    child.kill().expect("failed to kill child");
    let status = child.wait().expect("error waiting on pidfd");
    assert_eq!(status.signal(), Some(libc::SIGKILL));

    let _ = Command::new("echo")
        .create_pidfd(false)
        .spawn()
        .unwrap()
        .pidfd()
        .expect_err("pidfd should not have been created when create_pid(false) is set");

    let _ = Command::new("echo")
        .spawn()
        .unwrap()
        .pidfd()
        .expect_err("pidfd should not have been created");

    // exercise the fork/exec path since the earlier attempts may have used pidfd_spawnp()
    let mut child =
        unsafe { Command::new("false").pre_exec(|| Ok(())) }.create_pidfd(true).spawn().unwrap();

    assert!(child.id() > 0 && child.id() < -1i32 as u32);

    if pidfd_open_available {
        assert!(child.pidfd().is_ok())
    }
    child.wait().expect("error waiting on child");
}

#[test]
fn test_pidfd() {
    if !probe_pidfd_support() {
        return;
    }

    let child = Command::new("sleep")
        .arg("1000")
        .create_pidfd(true)
        .spawn()
        .expect("executing 'sleep' failed");

    let fd = child.into_pidfd().unwrap();

    assert_matches!(fd.try_wait(), Ok(None));
    fd.kill().expect("kill failed");
    fd.kill().expect("sending kill twice failed");
    let status = fd.wait().expect("1st wait failed");
    assert_eq!(status.signal(), Some(libc::SIGKILL));

    // Trying to wait again for a reaped child is safe since there's no pid-recycling race.
    // But doing so will return an error.
    let res = fd.wait();
    assert_matches!(res, Err(e) if e.raw_os_error() == Some(libc::ECHILD));

    // Ditto for additional attempts to kill an already-dead child.
    let res = fd.kill();
    assert_matches!(res, Err(e) if e.raw_os_error() == Some(libc::ESRCH));
}

fn probe_pidfd_support() -> bool {
    // pidfds require the pidfd_open syscall
    let our_pid = crate::process::id();
    let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, our_pid, 0) };
    if pidfd >= 0 {
        unsafe { libc::close(pidfd as RawFd) };
        true
    } else {
        false
    }
}
