// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See https://github.com/rust-lang/rust/issues/38942

#[repr(u64)]
pub enum NSEventType {
    NSEventTypePressure,
}

pub const A: u64 = NSEventType::NSEventTypePressure as u64;

fn banana() -> u64 {
    A
}

fn main() {
    println!("banana! {}", banana());
}
