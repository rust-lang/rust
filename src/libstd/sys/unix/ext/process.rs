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
use os::unix::io::{FromRawFd, RawFd, AsRawFd};
use prelude::v1::*;
use process;
use sys;
use sys_common::{AsInnerMut, AsInner, FromInner};

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
        match *self.as_inner() {
            sys::process::ExitStatus::Signal(s) => Some(s),
            _ => None
        }
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for process::Stdio {
    /// Creates a new instance of `Stdio` from the raw underlying file
    /// descriptor.
    ///
    /// When this `Stdio` is used as an I/O handle for a child process the given
    /// file descriptor will be `dup`d into the destination file descriptor in
    /// the child process.
    ///
    /// Note that this function **does not** take ownership of the file
    /// descriptor provided and it will **not** be closed when `Stdio` goes out
    /// of scope. As a result this method is unsafe because due to the lack of
    /// knowledge about the lifetime of the provided file descriptor, this could
    /// cause another I/O primitive's ownership property of its file descriptor
    /// to be violated.
    ///
    /// Also note that this file descriptor may be used multiple times to spawn
    /// processes. For example the `Command::spawn` function could be called
    /// more than once to spawn more than one process sharing this file
    /// descriptor.
    unsafe fn from_raw_fd(fd: RawFd) -> process::Stdio {
        process::Stdio::from_inner(sys::process::Stdio::Fd(fd))
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawFd for process::ChildStdin {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawFd for process::ChildStdout {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawFd for process::ChildStderr {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}
