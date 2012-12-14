// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
#[legacy_modes];

fn main() {
    // Make sure closing over can be a last use
    let q = ~10;
    let addr = ptr::addr_of(&(*q));
    let f = fn@(move q) -> *int { ptr::addr_of(&(*q)) };
    assert addr == f();

    // But only when it really is the last use
    let q = ~20;
    let f = fn@(copy q) -> *int { ptr::addr_of(&(*q)) };
    assert ptr::addr_of(&(*q)) != f();

    // Ensure function arguments and box arguments interact sanely.
    fn call_me(x: fn() -> int, y: ~int) { assert x() == *y; }
    let q = ~30;
    call_me({|copy q| *q}, q);

    // Check that no false positives are found in loops.
    let mut q = ~40, p = 10;
    loop {
        let i = copy q;
        p += *i;
        if p > 100 { break; }
    }

    // Verify that blocks can't interfere with each other.
    fn two_blocks(a: fn(), b: fn()) { a(); b(); a(); b(); }
    let q = ~50;
    two_blocks(|| { let a = q; assert *a == 50;},
               || { let a = q; assert *a == 50;});
}
