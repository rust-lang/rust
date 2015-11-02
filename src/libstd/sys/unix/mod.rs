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
#![allow(non_camel_case_types)]

use libc;
use num::One;
use ops::Neg;

#[cfg(target_os = "android")]   pub use os::android as platform;
#[cfg(target_os = "bitrig")]    pub use os::bitrig as platform;
#[cfg(target_os = "dragonfly")] pub use os::dragonfly as platform;
#[cfg(target_os = "freebsd")]   pub use os::freebsd as platform;
#[cfg(target_os = "ios")]       pub use os::ios as platform;
#[cfg(target_os = "linux")]     pub use os::linux as platform;
#[cfg(target_os = "macos")]     pub use os::macos as platform;
#[cfg(target_os = "nacl")]      pub use os::nacl as platform;
#[cfg(target_os = "netbsd")]    pub use os::netbsd as platform;
#[cfg(target_os = "openbsd")]   pub use os::openbsd as platform;

pub mod backtrace;

pub mod c;
pub mod condvar;
pub mod dynamic_lib;
pub mod fd;
pub mod fs;
pub mod mutex;
pub mod net;
pub mod raw;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod stack_overflow;
pub mod sync;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;
pub mod error;
pub mod env;
pub mod rt;
pub use sys::common::unwind;
pub use sys::common::os_str::u8 as os_str;

pub mod deps;
pub mod args;

pub use sys::common::c::cvt_neg1 as cvt;

pub fn cvt_r<T: One + PartialEq + Neg<Output=T>, F: FnMut() -> T>(mut f: F) -> error::Result<T> {
    loop {
        return match cvt(f()) {
            Err(ref e) if e.code() == libc::EINTR => continue,
            r => r,
        }
    }
}
