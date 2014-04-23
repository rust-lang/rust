// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1
// ignore-pretty

#![feature(phase)]

#[phase(syntax)]
extern crate fourcc;

static static_val: u32 = fourcc!("foo ");
static static_val_be: u32 = fourcc!("foo ", big);
static static_val_le: u32 = fourcc!("foo ", little);
static static_val_target: u32 = fourcc!("foo ", target);

fn main() {
    let val = fourcc!("foo ", big);
    assert_eq!(val, 0x666f6f20u32);
    assert_eq!(val, fourcc!("foo "));

    let val = fourcc!("foo ", little);
    assert_eq!(val, 0x206f6f66u32);

    let val = fourcc!("foo ", target);
    let exp = if cfg!(target_endian = "big") { 0x666f6f20u32 } else { 0x206f6f66u32 };
    assert_eq!(val, exp);

    assert_eq!(static_val_be, 0x666f6f20u32);
    assert_eq!(static_val, static_val_be);
    assert_eq!(static_val_le, 0x206f6f66u32);
    let exp = if cfg!(target_endian = "big") { 0x666f6f20u32 } else { 0x206f6f66u32 };
    assert_eq!(static_val_target, exp);

    assert_eq!(fourcc!("\xC0\xFF\xEE!"), 0xC0FFEE21);
}
