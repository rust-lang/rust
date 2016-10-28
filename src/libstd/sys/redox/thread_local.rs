// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // not used on all platforms

pub type Key = usize;

#[inline]
pub unsafe fn create(_dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    panic!("pthread key create not supported");
    //let mut key = 0;
    //assert_eq!(libc::pthread_key_create(&mut key, mem::transmute(dtor)), 0);
    //key
}

#[inline]
pub unsafe fn set(_key: Key, _value: *mut u8) {
    panic!("pthread key set not supported");
    //let r = libc::pthread_setspecific(key, value as *mut _);
    //debug_assert_eq!(r, 0);
}

#[inline]
pub unsafe fn get(_key: Key) -> *mut u8 {
    panic!("pthread key get not supported");
    //libc::pthread_getspecific(key) as *mut u8
}

#[inline]
pub unsafe fn destroy(_key: Key) {
    panic!("pthread key destroy not supported");
    //let r = libc::pthread_key_delete(key);
    //debug_assert_eq!(r, 0);
}
