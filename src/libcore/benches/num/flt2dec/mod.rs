// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod strategy {
    mod dragon;
    mod grisu;
}

use std::f64;
use std::io::Write;
use std::vec::Vec;
use test::Bencher;
use core::num::flt2dec::{decode, DecodableFloat, FullDecoded, Decoded};
use core::num::flt2dec::MAX_SIG_DIGITS;

pub fn decode_finite<T: DecodableFloat>(v: T) -> Decoded {
    match decode(v).1 {
        FullDecoded::Finite(decoded) => decoded,
        full_decoded => panic!("expected finite, got {:?} instead", full_decoded)
    }
}

#[bench]
fn bench_small_shortest(b: &mut Bencher) {
    let mut buf = Vec::with_capacity(20);

    b.iter(|| {
        buf.clear();
        write!(&mut buf, "{}", 3.1415926f64).unwrap()
    });
}

#[bench]
fn bench_big_shortest(b: &mut Bencher) {
    let mut buf = Vec::with_capacity(300);

    b.iter(|| {
        buf.clear();
        write!(&mut buf, "{}", f64::MAX).unwrap()
    });
}
