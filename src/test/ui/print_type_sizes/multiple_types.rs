// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z print-type-sizes

// This file illustrates that when multiple structural types occur in
// a function, every one of them is included in the output.

pub struct SevenBytes([u8;  7]);
pub struct FiftyBytes([u8; 50]);

pub enum Enum {
    Small(SevenBytes),
    Large(FiftyBytes),
}

pub fn main() {
    let _e: Enum;
    let _f: FiftyBytes;
    let _s: SevenBytes;
}
