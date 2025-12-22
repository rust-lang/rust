use crate::io;
use crate::os::fd::FromRawFd;
use crate::sys::fd::FileDesc;
use crate::sys::pal::cvt;

pub type Pipe = FileDesc;

pub fn pipe() -> io::Result<(Pipe, Pipe)> {
    let mut fds = [0; 2];

    // The only known way right now to create atomically set the CLOEXEC flag is
    // to use the `pipe2` syscall. This was added to Linux in 2.6.27, glibc 2.9
    // and musl 0.9.3, and some other targets also have it.
    cfg_select! {
        any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "hurd",
            target_os = "illumos",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "cygwin",
            target_os = "redox"
        ) => {
            unsafe {
                cvt(libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC))?;
                Ok((Pipe::from_raw_fd(fds[0]), Pipe::from_raw_fd(fds[1])))
            }
        }
        _ => {
            unsafe {
                cvt(libc::pipe(fds.as_mut_ptr()))?;

                let fd0 = Pipe::from_raw_fd(fds[0]);
                let fd1 = Pipe::from_raw_fd(fds[1]);
                fd0.set_cloexec()?;
                fd1.set_cloexec()?;
                Ok((fd0, fd1))
            }
        }
    }
}
