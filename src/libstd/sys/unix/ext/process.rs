// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific extensions to primitives in the `std::process` module.

#![stable(feature = "rust1", since = "1.0.0")]

use os::unix::raw::{uid_t, gid_t};
use os::unix::io::{FromRawFd, RawFd, AsRawFd, IntoRawFd};
use process;
use sys;
use sys_common::{AsInnerMut, AsInner, FromInner, IntoInner};

/// Unix-specific extensions to the `std::process::Command` builder
#[stable(feature = "rust1", since = "1.0.0")]
pub trait CommandExt {
    /// Sets the child process's user id. This translates to a
    /// `setuid` call in the child process. Failure in the `setuid`
    /// call will cause the spawn to fail.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn uid(&mut self, id: uid_t) -> &mut process::Command;

    /// Similar to `uid`, but sets the group id of the child process. This has
    /// the same semantics as the `uid` field.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn gid(&mut self, id: gid_t) -> &mut process::Command;

    /// Create a new session (cf. `setsid(2)`) for the child process. This means
    /// that the child is the leader of a new process group. The parent process
    /// remains the child reaper of the new process.
    ///
    /// If the tty argument is not `None`, this file will be the controlling
    /// terminal for the new session (cf. `login_tty(3)` and tty_ioctl(4),
    /// specifically the section on `TIOCSCTTY`). Note that this does not set
    /// that file to be the stdio handles for the child process, to do that you
    /// must set them directly with the `stdin()` et al methods.
    ///
    /// This is not enough to create a daemon process. The *init* process should
    /// be the child reaper of a daemon. This can be achieved if the parent
    /// process exit. Moreover, a daemon should not have a controlling terminal.
    #[unstable(feature = "process_session_leader", reason = "recently added",
               issue = "27811")]
    fn session_leader<T: AsRawFd>(&mut self, tty: Option<&T>) -> &mut process::Command;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl CommandExt for process::Command {
    fn uid(&mut self, id: uid_t) -> &mut process::Command {
        self.as_inner_mut().uid = Some(id);
        self
    }

    fn gid(&mut self, id: gid_t) -> &mut process::Command {
        self.as_inner_mut().gid = Some(id);
        self
    }

    fn session_leader<T: AsRawFd>(&mut self, tty: Option<&T>) -> &mut process::Command {
        {
            let inner = self.as_inner_mut();
            inner.session_leader = true;
            inner.tty = tty.map(T::as_raw_fd);
        }
        self
    }
}

/// Unix-specific extensions to `std::process::ExitStatus`
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExitStatusExt {
    /// If the process was terminated by a signal, returns that signal.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn signal(&self) -> Option<i32>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ExitStatusExt for process::ExitStatus {
    fn signal(&self) -> Option<i32> {
        self.as_inner().signal()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawFd for process::Stdio {
    unsafe fn from_raw_fd(fd: RawFd) -> process::Stdio {
        process::Stdio::from_inner(sys::fd::FileDesc::new(fd))
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdin {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdout {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStderr {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawFd for process::ChildStdin {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawFd for process::ChildStdout {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawFd for process::ChildStderr {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}
