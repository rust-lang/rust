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

use libc::funcs::posix01::signal::signal;
use error::prelude::*;
use libc;
use core::num::One;
use core::ops::Neg;

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

//#[cfg(not(feature = "disable-backtrace"))]
pub mod backtrace;

pub mod c;
pub mod condvar;
pub mod dynamic_lib;
pub mod ext;
pub mod fd;
pub mod fs;
pub mod mutex;
pub mod net;
//pub mod os;
//pub mod os_str;
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
//pub mod deps;

pub fn init() {
    // By default, some platforms will send a *signal* when an EPIPE error
    // would otherwise be delivered. This runtime doesn't install a SIGPIPE
    // handler, causing it to kill the program, which isn't exactly what we
    // want!
    //
    // Hence, we set SIGPIPE to ignore when the program starts up in order
    // to prevent this problem.
    unsafe {
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != !0);
    }
}

/*pub fn cvt<T: One + PartialEq + Neg<Output=T>>(t: T) -> err::Result<T> {
    let one: T = T::one();
    if t == -one {
        Err(err::imp::Error::last_error().expect("expected valid errno"))
    } else {
        Ok(t)
    }
}

pub fn cvt_r<T, F>(mut f: F) -> err::Result<T>
    where T: One + PartialEq + Neg<Output=T>, F: FnMut() -> T
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.code() == libc::EINTR => {}
            other => return other,
        }
    }
}*/

pub fn cvt_r<T: One + PartialEq + Neg<Output=T>, F: FnMut() -> T>(mut f: F) -> Result<T> {
    loop {
        return match f() {
            ref r if r == &-T::one() => match Error::expect_last_result() {
                Err(ref e) if e.code() == libc::EINTR => continue,
                e => e,
            },
            r => Ok(r),
        }
    }
}

pub fn page_size() -> usize {
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
}
