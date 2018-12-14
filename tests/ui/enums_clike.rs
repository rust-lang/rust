// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-x86

#![warn(clippy::all)]
#![allow(unused)]

#[repr(usize)]
enum NonPortable {
    X = 0x1_0000_0000,
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
}

enum NonPortableNoHint {
    X = 0x1_0000_0000,
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
}

#[repr(isize)]
enum NonPortableSigned {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF,
    A = 0x1_0000_0000,
    B = std::i32::MIN as isize,
    C = (std::i32::MIN as isize) - 1,
}

enum NonPortableSignedNoHint {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF,
    A = 0x1_0000_0000,
}

/*
FIXME: uncomment once https://github.com/rust-lang/rust/issues/31910 is fixed
#[repr(usize)]
enum NonPortable2<T: Trait> {
    X = Trait::Number,
    Y = 0,
}

trait Trait {
    const Number: usize = 0x1_0000_0000;
}
*/

fn main() {}
