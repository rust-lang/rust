use super::*;

use crate::ffi::OsStr;
use crate::mem;
use crate::ptr;
use crate::sys::cvt;

macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(t) => t,
            Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
        }
    };
}

#[test]
#[cfg_attr(
    any(
        // See #14232 for more information, but it appears that signal delivery to a
        // newly spawned process may just be raced in the macOS, so to prevent this
        // test from being flaky we ignore it on macOS.
        target_os = "macos",
        // When run under our current QEMU emulation test suite this test fails,
        // although the reason isn't very clear as to why. For now this test is
        // ignored there.
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv64",
    ),
    ignore
)]
fn test_process_mask() {
    unsafe {
        // Test to make sure that a signal mask does not get inherited.
        let mut cmd = Command::new(OsStr::new("cat"));

        let mut set = mem::MaybeUninit::<libc::sigset_t>::uninit();
        let mut old_set = mem::MaybeUninit::<libc::sigset_t>::uninit();
        t!(cvt(sigemptyset(set.as_mut_ptr())));
        t!(cvt(sigaddset(set.as_mut_ptr(), libc::SIGINT)));
        t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, set.as_ptr(), old_set.as_mut_ptr())));

        cmd.stdin(Stdio::MakePipe);
        cmd.stdout(Stdio::MakePipe);

        let (mut cat, mut pipes) = t!(cmd.spawn(Stdio::Null, true));
        let stdin_write = pipes.stdin.take().unwrap();
        let stdout_read = pipes.stdout.take().unwrap();

        t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, old_set.as_ptr(), ptr::null_mut())));

        t!(cvt(libc::kill(cat.id() as libc::pid_t, libc::SIGINT)));
        // We need to wait until SIGINT is definitely delivered. The
        // easiest way is to write something to cat, and try to read it
        // back: if SIGINT is unmasked, it'll get delivered when cat is
        // next scheduled.
        let _ = stdin_write.write(b"Hello");
        drop(stdin_write);

        // Either EOF or failure (EPIPE) is okay.
        let mut buf = [0; 5];
        if let Ok(ret) = stdout_read.read(&mut buf) {
            assert_eq!(ret, 0);
        }

        t!(cat.wait());
    }
}
