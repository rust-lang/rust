// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use libc::c_int;

pub type Key = pthread_key_t;

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key { unimplemented!() }

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) { unimplemented!() }

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 { unimplemented!() }

#[inline]
pub unsafe fn destroy(key: Key) { unimplemented!() }

#[cfg(target_os = "macos")]
type pthread_key_t = ::libc::c_ulong;

#[cfg(not(target_os = "macos"))]
type pthread_key_t = ::libc::c_uint;

extern {
    fn pthread_key_create(key: *mut pthread_key_t,
                          dtor: Option<unsafe extern fn(*mut u8)>) -> c_int;
    fn pthread_key_delete(key: pthread_key_t) -> c_int;
    fn pthread_getspecific(key: pthread_key_t) -> *mut u8;
    fn pthread_setspecific(key: pthread_key_t, value: *mut u8) -> c_int;
}
