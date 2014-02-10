// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// does the second one subsume the first?
// xfail-test
// xfail-fast

// notes on this test case:
// On Thu, Apr 18, 2013 at 6:30 PM, John Clements <clements@brinckerhoff.org> wrote:
// the "issue-2185.rs" test was xfailed with a ref to #2263. Issue #2263 is now fixed,
// so I tried it again, and after adding some &self parameters, I got this error:
//
// Running /usr/local/bin/rustc:
// issue-2185.rs:24:0: 26:1 error: conflicting implementations for a trait
// issue-2185.rs:24 impl iterable<uint> for 'static ||uint|| {
// issue-2185.rs:25     fn iter(&self, blk: |v: uint|) { self( |i| blk(i) ) }
// issue-2185.rs:26 }
// issue-2185.rs:20:0: 22:1 note: note conflicting implementation here
// issue-2185.rs:20 impl<A> iterable<A> for 'static ||A|| {
// issue-2185.rs:21     fn iter(&self, blk: |A|) { self(blk); }
// issue-2185.rs:22 }
//
// … so it looks like it's just not possible to implement both
// the generic iterable<uint> and iterable<A> for the type iterable<uint>.
// Is it okay if I just remove this test?
//
// but Niko responded:
// think it's fine to remove this test, just because it's old and cruft and not hard to reproduce.
// *However* it should eventually be possible to implement the same interface for the same type
// multiple times with different type parameters, it's just that our current trait implementation
// has accidental limitations.

// so I'm leaving it in.
// actually, it looks like this is related to bug #3429. I'll rename this bug.

// This test had to do with an outdated version of the iterable trait.
// However, the condition it was testing seemed complex enough to
// warrant still having a test, so I inlined the old definitions.

trait iterable<A> {
    fn iter(&self, blk: |A|);
}

impl<A> iterable<A> for 'static ||A|| {
    fn iter(&self, blk: |A|) { self(blk); }
}

impl iterable<uint> for 'static ||uint|| {
    fn iter(&self, blk: |v: uint|) { self( |i| blk(i) ) }
}

fn filter<A,IA:iterable<A>>(self: IA, prd: 'static |A| -> bool, blk: |A|) {
    self.iter(|a| {
        if prd(a) { blk(a) }
    });
}

fn foldl<A,B,IA:iterable<A>>(self: IA, b0: B, blk: |B, A| -> B) -> B {
    let mut b = b0;
    self.iter(|a| {
        b = blk(b, a);
    });
    b
}

fn range(lo: uint, hi: uint, it: |uint|) {
    let mut i = lo;
    while i < hi {
        it(i);
        i += 1u;
    }
}

pub fn main() {
    let range: 'static ||uint|| = |a| range(0u, 1000u, a);
    let filt: 'static ||v: uint|| = |a| filter(
        range,
        |&&n: uint| n % 3u != 0u && n % 5u != 0u,
        a);
    let sum = foldl(filt, 0u, |accum, &&n: uint| accum + n );

    println!("{}", sum);
}
