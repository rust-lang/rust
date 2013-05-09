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

// A trait for objects that can be used to do an if-then-else
// (No actual need for this to be static, but it is a simple test.)
trait bool_like {
    fn select<A>(b: Self, x1: A, x2: A) -> A;
}

fn andand<T:bool_like + Copy>(x1: T, x2: T) -> T {
    bool_like::select(x1, x2, x1)
}

impl bool_like for bool {
    fn select<A>(b: bool, x1: A, x2: A) -> A {
        if b { x1 } else { x2 }
    }
}

impl bool_like for int {
    fn select<A>(b: int, x1: A, x2: A) -> A {
        if b != 0 { x1 } else { x2 }
    }
}

// A trait for sequences that can be constructed imperatively.
trait buildable<A> {
     fn build_sized(size: uint, builder: &fn(push: &fn(v: A))) -> Self;
}


impl<A> buildable<A> for @[A] {
    #[inline(always)]
     fn build_sized(size: uint, builder: &fn(push: &fn(v: A))) -> @[A] {
         at_vec::build_sized(size, builder)
     }
}
impl<A> buildable<A> for ~[A] {
    #[inline(always)]
     fn build_sized(size: uint, builder: &fn(push: &fn(v: A))) -> ~[A] {
         vec::build_sized(size, builder)
     }
}

#[inline(always)]
fn build<A, B: buildable<A>>(builder: &fn(push: &fn(v: A))) -> B {
    buildable::build_sized(4, builder)
}

/// Apply a function to each element of an iterable and return the results
fn map<T, IT: BaseIter<T>, U, BU: buildable<U>>
    (v: IT, f: &fn(&T) -> U) -> BU {
    do build |push| {
        for v.each() |elem| {
            push(f(elem));
        }
    }
}

fn seq_range<BT:buildable<int>>(lo: uint, hi: uint) -> BT {
    do buildable::build_sized(hi-lo) |push| {
        for uint::range(lo, hi) |i| {
            push(i as int);
        }
    }
}

pub fn main() {
    let v: @[int] = seq_range(0, 10);
    assert!(v == @[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let v: @[int] = map(&[1,2,3], |&x| 1+x);
    assert!(v == @[2, 3, 4]);
    let v: ~[int] = map(&[1,2,3], |&x| 1+x);
    assert!(v == ~[2, 3, 4]);

    assert!(bool_like::select(true, 9, 14) == 9);
    assert!(!andand(true, false));
    assert!(andand(7, 12) == 12);
    assert!(andand(0, 12) == 0);
}
