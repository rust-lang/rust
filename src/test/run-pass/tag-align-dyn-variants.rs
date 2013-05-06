// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

enum a_tag<A,B> {
    varA(A),
    varB(B)
}

struct t_rec<A,B> {
    chA: u8,
    tA: a_tag<A,B>,
    chB: u8,
    tB: a_tag<A,B>
}

fn mk_rec<A:Copy,B:Copy>(a: A, b: B) -> t_rec<A,B> {
    return t_rec{ chA:0u8, tA:varA(a), chB:1u8, tB:varB(b) };
}

fn is_aligned<A>(amnt: uint, u: &A) -> bool {
    let p = ptr::to_unsafe_ptr(u) as uint;
    return (p & (amnt-1u)) == 0u;
}

fn variant_data_is_aligned<A,B>(amnt: uint, u: &a_tag<A,B>) -> bool {
    match u {
      &varA(ref a) => is_aligned(amnt, a),
      &varB(ref b) => is_aligned(amnt, b)
    }
}

pub fn main() {
    let x = mk_rec(22u64, 23u64);
    assert!(is_aligned(8u, &x.tA));
    assert!(variant_data_is_aligned(8u, &x.tA));
    assert!(is_aligned(8u, &x.tB));
    assert!(variant_data_is_aligned(8u, &x.tB));

    let x = mk_rec(22u64, 23u32);
    assert!(is_aligned(8u, &x.tA));
    assert!(variant_data_is_aligned(8u, &x.tA));
    assert!(is_aligned(8u, &x.tB));
    assert!(variant_data_is_aligned(4u, &x.tB));

    let x = mk_rec(22u32, 23u64);
    assert!(is_aligned(8u, &x.tA));
    assert!(variant_data_is_aligned(4u, &x.tA));
    assert!(is_aligned(8u, &x.tB));
    assert!(variant_data_is_aligned(8u, &x.tB));

    let x = mk_rec(22u32, 23u32);
    assert!(is_aligned(4u, &x.tA));
    assert!(variant_data_is_aligned(4u, &x.tA));
    assert!(is_aligned(4u, &x.tB));
    assert!(variant_data_is_aligned(4u, &x.tB));

    let x = mk_rec(22f64, 23f64);
    assert!(is_aligned(8u, &x.tA));
    assert!(variant_data_is_aligned(8u, &x.tA));
    assert!(is_aligned(8u, &x.tB));
    assert!(variant_data_is_aligned(8u, &x.tB));
}
