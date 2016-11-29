// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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
use sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};

pub type Key = usize;

type Dtor = unsafe extern fn(*mut u8);

static NEXT_KEY: AtomicUsize = ATOMIC_USIZE_INIT;

static mut KEYS: *mut BTreeMap<Key, Option<Dtor>> = ptr::null_mut();

#[thread_local]
static mut LOCALS: *mut BTreeMap<Key, *mut u8> = ptr::null_mut();

unsafe fn keys() -> &'static mut BTreeMap<Key, Option<Dtor>> {
    if KEYS == ptr::null_mut() {
        KEYS = Box::into_raw(Box::new(BTreeMap::new()));
    }
    &mut *KEYS
}

unsafe fn locals() -> &'static mut BTreeMap<Key, *mut u8> {
    if LOCALS == ptr::null_mut() {
        LOCALS = Box::into_raw(Box::new(BTreeMap::new()));
    }
    &mut *LOCALS
}

#[inline]
pub unsafe fn create(dtor: Option<Dtor>) -> Key {
    let key = NEXT_KEY.fetch_add(1, Ordering::SeqCst);
    keys().insert(key, dtor);
    key
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    if let Some(&entry) = locals().get(&key) {
        entry
    } else {
        ptr::null_mut()
    }
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    locals().insert(key, value);
}

#[inline]
pub unsafe fn destroy(key: Key) {
    keys().remove(&key);
}
