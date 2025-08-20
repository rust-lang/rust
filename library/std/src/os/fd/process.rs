use libc::{F_GETFD, F_SETFD, FD_CLOEXEC};

use crate::os::fd::raw::AsRawFd;
use crate::os::fd::{OwnedFd, RawFd};
use crate::process::Command;
use crate::sealed::Sealed;
use crate::sys::{cvt, cvt_r};
use crate::sys_common::AsInnerMut;

/// Extensions to the [`crate::process::Command`] builder for Unix and WASI, platforms that support file
/// descriptors.
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[unstable(feature = "command_pass_fds", issue = "144989")]
pub trait CommandExt: Sealed {
    /// Pass a file descriptor to a child process.
    ///
    /// Getting this right is tricky. It is recommended to provide further information to the child
    /// process by some other mechanism. This could be an argument confirming file descriptors that
    /// the child can use, device/inode numbers to allow for sanity checks, or something similar.
    ///
    /// If `new_fd` is an open file descriptor and closing it would produce one or more errors,
    /// those errors will be lost when this function is called. See
    /// [`man 2 dup`](https://www.man7.org/linux/man-pages/man2/dup.2.html#NOTES) for more information.
    ///
    /// If this method is called multiple times with the same `new_fd`, all but one file descriptor
    /// will be lost.
    ///
    /// ```
    /// #![feature(command_pass_fds)]
    ///
    /// use std::fs::OpenOptions;
    /// use std::process::Command;
    /// use std::os::fd::process::CommandExt;
    ///
    /// eprintln!("chom");
    /// let file = OpenOptions::new().read(true).write(true).create(true).open("/tmp/fd_doctest.rs").unwrap();
    /// eprintln!("chom");
    ///
    /// let mut command = Command::new("ls");
    /// command.arg("/proc/self/fd").fd(5, file);
    /// eprintln!("chom");
    ///
    /// let output = command.output().inspect(|o| eprintln!("{o:?}")).unwrap();
    /// assert_eq!(String::from_utf8_lossy(&output.stdout).split_whitespace().map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>(), vec![0,1,2,3,5]);
    /// ```
    fn fd(&mut self, new_fd: RawFd, old_fd: impl Into<OwnedFd>) -> &mut Self;
}

#[unstable(feature = "command_pass_fds", issue = "144989")]
impl CommandExt for Command {
    fn fd(&mut self, new_fd: RawFd, old_fd: impl Into<OwnedFd>) -> &mut Self {
        let old = old_fd.into().as_raw_fd();
        unsafe {
            self.as_inner_mut().pre_exec(Box::new(move || {
                cvt_r(|| libc::dup2(old, new_fd))?;
                let flags = cvt(libc::fcntl(new_fd, F_GETFD))?;
                cvt(libc::fcntl(new_fd, F_SETFD, flags & !FD_CLOEXEC))?;
                cvt_r(|| libc::close(old))?;
                Ok(())
            }))
        }

        self
    }
}
