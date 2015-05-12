// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Extensions to `std::process` for Windows.

#![stable(feature = "from_raw_os", since = "1.1.0")]

use os::windows::io::{FromRawHandle, RawHandle, AsRawHandle};
use process;
use sys;
use sys_common::{AsInner, FromInner};

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawHandle for process::Stdio {
    /// Creates a new instance of `Stdio` from the raw underlying handle.
    ///
    /// When this `Stdio` is used as an I/O handle for a child process the given
    /// handle will be duplicated via `DuplicateHandle` to ensure that the
    /// handle has the correct permissions to cross the process boundary.
    ///
    /// Note that this function **does not** take ownership of the handle
    /// provided and it will **not** be closed when `Stdio` goes out of scope.
    /// As a result this method is unsafe because due to the lack of knowledge
    /// about the lifetime of the provided handle, this could cause another I/O
    /// primitive's ownership property of its handle to be violated.
    ///
    /// Also note that this handle may be used multiple times to spawn
    /// processes. For example the `Command::spawn` function could be called
    /// more than once to spawn more than one process sharing this handle.
    unsafe fn from_raw_handle(handle: RawHandle) -> process::Stdio {
        let handle = sys::handle::RawHandle::new(handle as *mut _);
        process::Stdio::from_inner(sys::process::Stdio::Handle(handle))
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawHandle for process::Child {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawHandle for process::ChildStdin {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawHandle for process::ChildStdout {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl AsRawHandle for process::ChildStderr {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}
