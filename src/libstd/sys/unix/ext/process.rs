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

use collections::HashSet;
use os::unix::raw::{uid_t, gid_t};
use os::unix::io::{FromRawFd, RawFd, AsRawFd, IntoRawFd};
#[cfg(stage0)]
use prelude::v1::*;
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

    /// Create a new session (cf. `setsid(2)`) for the child process. This means that the child is
    /// the leader of a new process group. The parent process remains the child reaper of the new
    /// process.
    ///
    /// This is not enough to create a daemon process. The *init* process should be the child
    /// reaper of a daemon. This can be achieved if the parent process exit. Moreover, a daemon
    /// should not have a controlling terminal. To acheive this, a session leader (the child) must
    /// spawn another process (the daemon) in the same session.
    #[unstable(feature = "process_session_leader", reason = "recently added")]
    fn session_leader(&mut self, on: bool) -> &mut process::Command;

    /// Set to `false` to prevent file descriptors leak (default is `true`).
    #[unstable(feature = "process_leak_fds", reason = "recently added")]
    fn leak_fds(&mut self, on: bool) -> &mut process::Command;

    /// Allow to prevent file descriptors leak except for an authorized whitelist.
    ///
    /// The file descriptors in the whitelist will leak through *all* the subsequent executions
    /// (cf. `open(2)` and `O_CLOEXEC`). The new process should change the property of this file
    /// descriptors to avoid unintended leaks (cf. `fcntl(2)` and `FD_CLOEXEC`).
    #[unstable(feature = "process_leak_fds", reason = "recently added")]
    fn leak_fds_whitelist(&mut self, leak: HashSet<RawFd>) -> &mut process::Command;
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

    fn session_leader(&mut self, on: bool) -> &mut process::Command {
        self.as_inner_mut().session_leader = on;
        self
    }

    fn leak_fds(&mut self, on: bool) -> &mut process::Command {
        self.as_inner_mut().leak_fds = on;
        self
    }

    fn leak_fds_whitelist(&mut self, whitelist: HashSet<RawFd>) -> &mut process::Command {
        self.as_inner_mut().leak_fds_whitelist = whitelist;
        // Do not leak any FDs except those from the whitelist
        self.as_inner_mut().leak_fds = false;
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

impl IntoRawFd for process::ChildStdin {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

impl IntoRawFd for process::ChildStdout {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

impl IntoRawFd for process::ChildStderr {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}
