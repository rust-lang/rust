// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub type Key = usize;

pub unsafe fn create(_dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    panic!("TLS on wasm with atomics not implemented yet");
}

pub unsafe fn set(_key: Key, _value: *mut u8) {
    panic!("TLS on wasm with atomics not implemented yet");
}

pub unsafe fn get(_key: Key) -> *mut u8 {
    panic!("TLS on wasm with atomics not implemented yet");
}

pub unsafe fn destroy(_key: Key) {
    panic!("TLS on wasm with atomics not implemented yet");
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
