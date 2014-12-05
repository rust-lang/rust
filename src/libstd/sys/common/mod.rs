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
#![allow(dead_code)]

use io::{mod, IoError, IoResult};
use prelude::*;
use sys::{last_error, retry};
use c_str::CString;
use num::Int;
use path::BytesContainer;
use collections;

pub mod condvar;
pub mod helper_thread;
pub mod mutex;
pub mod net;
pub mod rwlock;
pub mod thread_local;

// common error constructors

pub fn eof() -> IoError {
    IoError {
        kind: io::EndOfFile,
        desc: "end of file",
        detail: None,
    }
}

pub fn timeout(desc: &'static str) -> IoError {
    IoError {
        kind: io::TimedOut,
        desc: desc,
        detail: None,
    }
}

pub fn short_write(n: uint, desc: &'static str) -> IoError {
    IoError {
        kind: if n == 0 { io::TimedOut } else { io::ShortWrite(n) },
        desc: desc,
        detail: None,
    }
}

pub fn unimpl() -> IoError {
    IoError {
        kind: io::IoUnavailable,
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

pub fn keep_going(data: &[u8], f: |*const u8, uint| -> i64) -> i64 {
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

// A trait for extracting representations from std::io types
pub trait AsInner<Inner> {
    fn as_inner(&self) -> &Inner;
}

pub trait ProcessConfig<K: BytesContainer, V: BytesContainer> {
    fn program(&self) -> &CString;
    fn args(&self) -> &[CString];
    fn env(&self) -> Option<&collections::HashMap<K, V>>;
    fn cwd(&self) -> Option<&CString>;
    fn uid(&self) -> Option<uint>;
    fn gid(&self) -> Option<uint>;
    fn detach(&self) -> bool;
}
