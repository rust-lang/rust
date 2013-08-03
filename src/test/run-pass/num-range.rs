// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::int;
use std::uint;

fn uint_range(lo: uint, hi: uint, it: &fn(uint) -> bool) -> bool {
    range(lo, hi).advance(it)
}

fn int_range(lo: int, hi: int, it: &fn(int) -> bool) -> bool {
    range(lo, hi).advance(it)
}

fn int_range_step(a: int, b: int, step: int, it: &fn(int) -> bool) -> bool {
    int::range_step(a, b, step, it)
}

fn uint_range_step(a: uint, b: uint, s: int, it: &fn(uint) -> bool) -> bool {
    uint::range_step(a, b, s, it)
}

pub fn main() {
    println("num-range start");
    // int and uint have same result for
    //   Sum{2 <= i < 100} == (Sum{1 <= i <= 99} - 1) == n*(n+1)/2 - 1 for n=99
    let mut sum = 0u;
    do uint_range(2, 100) |i| {
        sum += i;
        true
    };
    assert_eq!(sum, 4949);

    let mut sum = 0i;
    do int_range(2, 100) |i| {
        sum += i;
        true
    };
    assert_eq!(sum, 4949);


    // elements are visited in correct order
    let primes = [2,3,5,7];
    let mut prod = 1i;
    do uint_range(0, 4) |i| {
        prod *= int::pow(primes[i], i);
        true
    };
    assert_eq!(prod, 1*3*5*5*7*7*7);
    let mut prod = 1i;
    do int_range(0, 4) |i| {
        prod *= int::pow(primes[i], i as uint);
        true
    };
    assert_eq!(prod, 1*3*5*5*7*7*7);


    // empty ranges
    do int_range(10, 10) |_| {
        fail!("range should be empty when start == stop");
        true
    };

    do uint_range(10, 10) |_| {
        fail!("range should be empty when start == stop");
        true
    };


    // range iterations do not wrap/overflow
    let mut oflo_loop_visited = ~[];
    do uint_range_step(uint::max_value-15, uint::max_value, 4) |x| {
        oflo_loop_visited.push(uint::max_value - x);
        true
    };
    assert_eq!(oflo_loop_visited, ~[15, 11, 7, 3]);

    let mut oflo_loop_visited = ~[];
    do int_range_step(int::max_value-15, int::max_value, 4) |x| {
        oflo_loop_visited.push(int::max_value - x);
        true
    };
    assert_eq!(oflo_loop_visited, ~[15, 11, 7, 3]);


    // range_step never passes nor visits the stop element
    do int_range_step(0, 21, 3) |x| {
        assert!(x < 21);
        true
    };

    // range_step_inclusive will never pass stop element, and may skip it.
    let mut saw21 = false;
    do uint::range_step_inclusive(0, 21, 4) |x| {
        assert!(x <= 21);
        if x == 21 { saw21 = true; }
        true
    };
    assert!(!saw21);
    let mut saw21 = false;
    do int::range_step_inclusive(0, 21, 4) |x| {
        assert!(x <= 21);
        if x == 21 { saw21 = true; }
        true
    };
    assert!(!saw21);

    // range_step_inclusive will never pass stop element, but may visit it.
    let mut saw21 = false;
    do uint::range_step_inclusive(0, 21, 3) |x| {
        assert!(x <= 21);
        printfln!("saw: %u", x);
        if x == 21 { saw21 = true; }
        true
    };
    assert!(saw21);
    let mut saw21 = false;
    do int::range_step_inclusive(0, 21, 3) |x| {
        assert!(x <= 21);
        if x == 21 { saw21 = true; }
        true
    };
    assert!(saw21);

}
