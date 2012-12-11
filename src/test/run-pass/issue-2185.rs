// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test FIXME #2263
// xfail-fast
// This test had to do with an outdated version of the iterable trait.
// However, the condition it was testing seemed complex enough to
// warrant still having a test, so I inlined the old definitions.

#[legacy_modes];

trait iterable<A> {
    fn iter(blk: fn(A));
}

impl<A> fn@(fn(A)): iterable<A> {
    fn iter(blk: fn(A)) { self(blk); }
}

impl fn@(fn(uint)): iterable<uint> {
    fn iter(blk: fn(&&v: uint)) { self( |i| blk(i) ) }
}

fn filter<A,IA:iterable<A>>(self: IA, prd: fn@(A) -> bool, blk: fn(A)) {
    do self.iter |a| {
        if prd(a) { blk(a) }
    }
}

fn foldl<A,B,IA:iterable<A>>(self: IA, +b0: B, blk: fn(B, A) -> B) -> B {
    let mut b = move b0;
    do self.iter |a| {
        b = move blk(b, a);
    }
    move b
}

fn range(lo: uint, hi: uint, it: fn(uint)) {
    let mut i = lo;
    while i < hi {
        it(i);
        i += 1u;
    }
}

fn main() {
    let range: fn@(fn&(uint)) = |a| range(0u, 1000u, a);
    let filt: fn@(fn&(&&v: uint)) = |a| filter(
        range,
        |&&n: uint| n % 3u != 0u && n % 5u != 0u,
        a);
    let sum = foldl(filt, 0u, |accum, &&n: uint| accum + n );

    io::println(fmt!("%u", sum));
}
