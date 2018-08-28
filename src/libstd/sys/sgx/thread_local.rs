// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::abi::tls::{Tls, Key as AbiKey};

pub type Key = usize;

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    Tls::create(dtor).as_usize()
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    Tls::set(AbiKey::from_usize(key), value)
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    Tls::get(AbiKey::from_usize(key))
}

#[inline]
pub unsafe fn destroy(key: Key) {
    Tls::destroy(AbiKey::from_usize(key))
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
