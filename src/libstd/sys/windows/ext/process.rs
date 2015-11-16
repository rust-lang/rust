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

#![stable(feature = "process_extensions", since = "1.2.0")]

use os::windows::io::{FromRawHandle, RawHandle, AsRawHandle, IntoRawHandle};
use process;
use sys;
use sys_common::{AsInner, FromInner, IntoInner};

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawHandle for process::Stdio {
    unsafe fn from_raw_handle(handle: RawHandle) -> process::Stdio {
        let handle = sys::handle::Handle::new(handle as *mut _);
        process::Stdio::from_inner(handle)
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::Child {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawHandle for process::Child {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdin {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdout {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStderr {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawHandle for process::ChildStdin {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawHandle for process::ChildStdout {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl IntoRawHandle for process::ChildStderr {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}
