// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std;
use std::rand;
use uint::range;

// random uint less than n
fn under(r : rand::rng, n : uint) -> uint {
    assert!(n != 0u); r.next() as uint % n
}

// random choice from a vec
fn choice<T:copy>(r : rand::rng, v : ~[const T]) -> T {
    assert!(v.len() != 0u); v[under(r, v.len())]
}

// k in n chance of being true
fn likelihood(r : rand::rng, k : uint, n : uint) -> bool { under(r, n) < k }


static iters : uint = 1000u;
static vlen  : uint = 100u;

enum maybe_pointy {
    none,
    p(@pointy)
}

type pointy = {
    mut a : maybe_pointy,
    mut b : ~maybe_pointy,
    mut c : @maybe_pointy,

    mut f : @fn()->(),
    mut g : ~fn()->(),

    mut m : ~[maybe_pointy],
    mut n : ~[maybe_pointy],
    mut o : {x : int, y : maybe_pointy}
};
// To add: objects; traits; anything type-parameterized?

fn empty_pointy() -> @pointy {
    return @{
        mut a : none,
        mut b : ~none,
        mut c : @none,

        mut f : || {},
        mut g : || {},

        mut m : ~[],
        mut n : ~[],
        mut o : {x : 0, y : none}
    }
}

fn nopP(_x : @pointy) { }
fn nop<T>(_x: T) { }

fn test_cycles(r : rand::rng, k: uint, n: uint)
{
    let mut v : ~[@pointy] = ~[];

    // Create a graph with no edges
    range(0u, vlen) {|_i|
        v.push(empty_pointy());
    }

    // Fill in the graph with random edges, with density k/n
    range(0u, vlen) {|i|
        if (likelihood(r, k, n)) { v[i].a = p(choice(r, v)); }
        if (likelihood(r, k, n)) { v[i].b = ~p(choice(r, v)); }
        if (likelihood(r, k, n)) { v[i].c = @p(choice(r, v)); }

        if (likelihood(r, k, n)) { v[i].f = bind nopP(choice(r, v)); }
        //if (false)               { v[i].g = bind (|_: @pointy| { })(
        // choice(r, v)); }
          // https://github.com/mozilla/rust/issues/1899

        if (likelihood(r, k, n)) { v[i].m = [p(choice(r, v))]; }
        if (likelihood(r, k, n)) { v[i].n.push(mut p(choice(r, v))); }
        if (likelihood(r, k, n)) { v[i].o = {x: 0, y: p(choice(r, v))}; }
    }

    // Drop refs one at a time
    range(0u, vlen) {|i|
        v[i] = empty_pointy()
    }
}

fn main()
{
    let r = rand::rng();
    range(0u, iters) {|i|
        test_cycles(r, i, iters);
    }
}
