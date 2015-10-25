// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // sys isn't exported yet

use libc::c_int;

pub type Key = pthread_key_t;

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    let mut key = 0;
    assert_eq!(pthread_key_create(&mut key, dtor), 0);
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let r = pthread_setspecific(key, value);
    debug_assert_eq!(r, 0);
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    pthread_getspecific(key)
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let r = pthread_key_delete(key);
    debug_assert_eq!(r, 0);
}

#[cfg(any(target_os = "macos",
          target_os = "ios"))]
type pthread_key_t = ::libc::c_ulong;

#[cfg(any(target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "nacl"))]
type pthread_key_t = ::libc::c_int;

#[cfg(not(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "netbsd",
              target_os = "openbsd",
              target_os = "nacl")))]
type pthread_key_t = ::libc::c_uint;

extern {
    fn pthread_key_create(key: *mut pthread_key_t,
                          dtor: Option<unsafe extern fn(*mut u8)>) -> c_int;
    fn pthread_key_delete(key: pthread_key_t) -> c_int;
    fn pthread_getspecific(key: pthread_key_t) -> *mut u8;
    fn pthread_setspecific(key: pthread_key_t, value: *mut u8) -> c_int;
}
