// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test newsched transition

// This is testing for stack frames greater than 256 bytes,
// for which function prologues are generated differently

struct Biggy {
    a00: u64,
    a01: u64,
    a02: u64,
    a03: u64,
    a04: u64,
    a05: u64,
    a06: u64,
    a07: u64,
    a08: u64,
    a09: u64,
    a10: u64,
    a11: u64,
    a12: u64,
    a13: u64,
    a14: u64,
    a15: u64,
    a16: u64,
    a17: u64,
    a18: u64,
    a19: u64,
    a20: u64,
    a21: u64,
    a22: u64,
    a23: u64,
    a24: u64,
    a25: u64,
    a26: u64,
    a27: u64,
    a28: u64,
    a29: u64,
    a30: u64,
    a31: u64,
    a32: u64,
    a33: u64,
    a34: u64,
    a35: u64,
    a36: u64,
    a37: u64,
    a38: u64,
    a39: u64,
}


fn getbig(i: Biggy) {
    if i.a00 != 0u64 {
        getbig(Biggy{a00: i.a00 - 1u64,.. i});
    }
}

pub fn main() {
    getbig(Biggy {
        a00: 10000u64,
        a01: 10000u64,
        a02: 10000u64,
        a03: 10000u64,
        a04: 10000u64,
        a05: 10000u64,
        a06: 10000u64,
        a07: 10000u64,
        a08: 10000u64,
        a09: 10000u64,
        a10: 10000u64,
        a11: 10000u64,
        a12: 10000u64,
        a13: 10000u64,
        a14: 10000u64,
        a15: 10000u64,
        a16: 10000u64,
        a17: 10000u64,
        a18: 10000u64,
        a19: 10000u64,
        a20: 10000u64,
        a21: 10000u64,
        a22: 10000u64,
        a23: 10000u64,
        a24: 10000u64,
        a25: 10000u64,
        a26: 10000u64,
        a27: 10000u64,
        a28: 10000u64,
        a29: 10000u64,
        a30: 10000u64,
        a31: 10000u64,
        a32: 10000u64,
        a33: 10000u64,
        a34: 10000u64,
        a35: 10000u64,
        a36: 10000u64,
        a37: 10000u64,
        a38: 10000u64,
        a39: 10000u64,
    });
}
