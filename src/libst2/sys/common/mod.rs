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
use sys::{last_error, retry, fs};
use c_str::CString;
use num::Int;
use path::BytesContainer;
use collections;

pub mod net;
pub mod helper_thread;
pub mod thread_local;

// common error constructors

pub fn eof() -> IoError { unimplemented!() }

pub fn timeout(desc: &'static str) -> IoError { unimplemented!() }

pub fn short_write(n: uint, desc: &'static str) -> IoError { unimplemented!() }

pub fn unimpl() -> IoError { unimplemented!() }

// unix has nonzero values as errors
pub fn mkerr_libc<T: Int>(ret: T) -> IoResult<()> { unimplemented!() }

pub fn keep_going(data: &[u8], f: |*const u8, uint| -> i64) -> i64 { unimplemented!() }

// traits for extracting representations from

pub trait AsFileDesc {
    fn as_fd(&self) -> &fs::FileDesc;
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
