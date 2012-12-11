// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Note: the borrowck analysis is currently flow-insensitive.
// Therefore, some of these errors are marked as spurious and could be
// corrected by a simple change to the analysis.  The others are
// either genuine or would require more advanced changes.  The latter
// cases are noted.

fn borrow(_v: &int) {}

fn inc(v: &mut ~int) {
    *v = ~(**v + 1);
}

fn post_aliased_const() {
    let mut v = ~3;
    borrow(v);
    let _w = &const v;
}

fn post_aliased_mut() {
    // SPURIOUS--flow
    let mut v = ~3;
    borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
    let _w = &mut v; //~ NOTE prior loan as mutable granted here
}

fn post_aliased_scope(cond: bool) {
    let mut v = ~3;
    borrow(v);
    if cond { inc(&mut v); }
}

fn loop_overarching_alias_mut() {
    let mut v = ~3;
    let mut _x = &mut v; //~ NOTE prior loan as mutable granted here
    loop {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
    }
}

fn block_overarching_alias_mut() {
    let mut v = ~3;
    let mut _x = &mut v; //~ NOTE prior loan as mutable granted here
    for 3.times {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
    }
}

fn loop_aliased_mut() {
    let mut v = ~3, w = ~4;
    let mut _x = &mut w;
    loop {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
        _x = &mut v; //~ NOTE prior loan as mutable granted here
    }
}

fn while_aliased_mut(cond: bool) {
    let mut v = ~3, w = ~4;
    let mut _x = &mut w;
    while cond {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
        _x = &mut v; //~ NOTE prior loan as mutable granted here
    }
}

fn while_aliased_mut_cond(cond: bool, cond2: bool) {
    let mut v = ~3, w = ~4;
    let mut _x = &mut w;
    while cond {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
        if cond2 {
            _x = &mut v; //~ NOTE prior loan as mutable granted here
        }
    }
}

fn loop_in_block() {
    let mut v = ~3, w = ~4;
    let mut _x = &mut w;
    for uint::range(0u, 10u) |_i| {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
        _x = &mut v; //~ NOTE prior loan as mutable granted here
    }
}

fn at_most_once_block() {
    fn at_most_once(f: fn()) { f() }

    // Here, the borrow check has no way of knowing that the block is
    // executed at most once.

    let mut v = ~3, w = ~4;
    let mut _x = &mut w;
    do at_most_once {
        borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
        _x = &mut v; //~ NOTE prior loan as mutable granted here
    }
}

fn main() {}
