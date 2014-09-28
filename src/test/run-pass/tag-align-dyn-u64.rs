// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-linux #7340 fails on 32-bit Linux
// ignore-macos #7340 fails on 32-bit macos

use std::mem;

enum Tag<A> {
    Tag2(A)
}

struct Rec {
    c8: u8,
    t: Tag<u64>
}

fn mk_rec() -> Rec {
    return Rec { c8:0u8, t:Tag2(0u64) };
}

fn is_8_byte_aligned(u: &Tag<u64>) -> bool {
    let p: uint = unsafe { mem::transmute(u) };
    return (p & 7u) == 0u;
}

pub fn main() {
    let x = mk_rec();
    assert!(is_8_byte_aligned(&x.t));
}
