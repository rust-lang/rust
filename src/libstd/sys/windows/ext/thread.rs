// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Extensions to `std::thread` for Windows.

#![unstable(feature = "thread_extensions", issue = "29791")]

use os::windows::io::{RawHandle, AsRawHandle, IntoRawHandle};
use thread;
use sys_common::{AsInner, IntoInner};

impl<T> AsRawHandle for thread::JoinHandle<T> {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

impl<T> IntoRawHandle for thread::JoinHandle<T>  {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}
