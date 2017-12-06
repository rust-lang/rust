// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use boxed::Box;
use ptr;

pub type Key = usize;

struct Allocated {
    value: *mut u8,
    dtor: Option<unsafe extern fn(*mut u8)>,
}

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    Box::into_raw(Box::new(Allocated {
        value: ptr::null_mut(),
        dtor,
    })) as usize
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    (*(key as *mut Allocated)).value = value;
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    (*(key as *mut Allocated)).value
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let key = Box::from_raw(key as *mut Allocated);
    if let Some(f) = key.dtor {
        f(key.value);
    }
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
