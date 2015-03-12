// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]

use old_io::{self, IoError, IoResult};
use prelude::v1::*;
use sys::{last_error, retry};
use ffi::CString;
use num::Int;
use old_path::BytesContainer;
use collections;

#[macro_use] pub mod helper_thread;

pub mod backtrace;
pub mod condvar;
pub mod mutex;
pub mod net;
pub mod net2;
pub mod rwlock;
pub mod stack;
pub mod thread;
pub mod thread_info;
pub mod thread_local;
pub mod wtf8;

// common error constructors

pub fn eof() -> IoError {
    IoError {
        kind: old_io::EndOfFile,
        desc: "end of file",
        detail: None,
    }
}

pub fn timeout(desc: &'static str) -> IoError {
    IoError {
        kind: old_io::TimedOut,
        desc: desc,
        detail: None,
    }
}

pub fn short_write(n: uint, desc: &'static str) -> IoError {
    IoError {
        kind: if n == 0 { old_io::TimedOut } else { old_io::ShortWrite(n) },
        desc: desc,
        detail: None,
    }
}

pub fn unimpl() -> IoError {
    IoError {
        kind: old_io::IoUnavailable,
        desc: "operations not yet supported",
        detail: None,
    }
}

// unix has nonzero values as errors
pub fn mkerr_libc<T: Int>(ret: T) -> IoResult<()> {
    if ret != Int::zero() {
        Err(last_error())
    } else {
        Ok(())
    }
}

pub fn keep_going<F>(data: &[u8], mut f: F) -> i64 where
    F: FnMut(*const u8, uint) -> i64,
{
    let origamt = data.len();
    let mut data = data.as_ptr();
    let mut amt = origamt;
    while amt > 0 {
        let ret = retry(|| f(data, amt));
        if ret == 0 {
            break
        } else if ret != -1 {
            amt -= ret as uint;
            data = unsafe { data.offset(ret as int) };
        } else {
            return ret;
        }
    }
    return (origamt - amt) as i64;
}

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types
#[doc(hidden)]
pub trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations
#[doc(hidden)]
pub trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}

#[doc(hidden)]
pub trait ProcessConfig<K: BytesContainer, V: BytesContainer> {
    fn program(&self) -> &CString;
    fn args(&self) -> &[CString];
    fn env(&self) -> Option<&collections::HashMap<K, V>>;
    fn cwd(&self) -> Option<&CString>;
    fn uid(&self) -> Option<uint>;
    fn gid(&self) -> Option<uint>;
    fn detach(&self) -> bool;
}
