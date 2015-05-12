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
use prelude::v1::*;
use process;
use sys;
use sys_common::{AsInnerMut, AsInner};

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
