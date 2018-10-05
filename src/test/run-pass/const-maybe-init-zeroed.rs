// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(maybe_uninit, const_maybe_uninit_zeroed)]

use std::mem;

fn main() {
    const UNIT: mem::MaybeUninit<()> = mem::MaybeUninit::zeroed();
    let bytes: [u8; 0] = unsafe { mem::transmute(UNIT) };
    assert_eq!(bytes, [0u8; 0]);

    const STRING: mem::MaybeUninit<String> = mem::MaybeUninit::zeroed();
    let bytes: [u8; mem::size_of::<String>()] = unsafe { mem::transmute(STRING) };
    assert_eq!(bytes, [0u8; mem::size_of::<String>()]);

    const U8: mem::MaybeUninit<u8> = mem::MaybeUninit::zeroed();
    let bytes: [u8; 1] = unsafe { mem::transmute(U8) };
    assert_eq!(bytes, [0u8; 1]);
}
