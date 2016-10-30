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

use collections::BTreeMap;
use ptr;

pub type Key = usize;

type Dtor = unsafe extern fn(*mut u8);

#[thread_local]
static mut NEXT_KEY: Key = 0;

#[thread_local]
static mut LOCALS: *mut BTreeMap<Key, (*mut u8, Option<Dtor>)> = ptr::null_mut();

unsafe fn locals() -> &'static mut BTreeMap<Key, (*mut u8, Option<Dtor>)> {
    if LOCALS == ptr::null_mut() {
        LOCALS = Box::into_raw(Box::new(BTreeMap::new()));
    }
    &mut *LOCALS
}

#[inline]
pub unsafe fn create(dtor: Option<Dtor>) -> Key {
    let key = NEXT_KEY;
    NEXT_KEY += 1;
    locals().insert(key, (0 as *mut u8, dtor));
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    locals().get_mut(&key).unwrap().0 = value;
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    locals()[&key].0
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let (value, dtor) = locals().remove(&key).unwrap();
    if let Some(dtor_fn) = dtor {
        dtor_fn(value);
    }
}
