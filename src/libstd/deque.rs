// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::container::{Container, Mutable};
use core::prelude::*;
use core::vec;

const initial_capacity: uint = 32u; // 2^5

pub struct Deque<T> {
    priv nelts: uint,
    priv lo: uint,
    priv hi: uint,
    priv elts: ~[Option<T>]
}

impl<T> Container for Deque<T> {
    pure fn len(&self) -> uint { self.nelts }
    pure fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<T> Mutable for Deque<T> {
    fn clear(&mut self) {
        for vec::each_mut(self.elts) |x| { *x = None }
        self.nelts = 0;
        self.lo = 0;
        self.hi = 0;
    }
}

pub impl<T> Deque<T> {
    static pure fn new() -> Deque<T> {
        Deque{nelts: 0, lo: 0, hi: 0,
              elts: vec::from_fn(initial_capacity, |_| None)}
    }

    fn peek_front(&self) -> &self/T { get(self.elts, self.lo) }
    fn peek_back(&self) -> &self/T { get(self.elts, self.hi - 1u) }

    fn get(&self, i: int) -> &self/T {
        let idx = (self.lo + (i as uint)) % self.elts.len();
        get(self.elts, idx)
    }

    fn pop_front(&mut self) -> T {
        let mut result = self.elts[self.lo].swap_unwrap();
        self.lo = (self.lo + 1u) % self.elts.len();
        self.nelts -= 1u;
        result
    }

    fn pop_back(&mut self) -> T {
        if self.hi == 0u {
            self.hi = self.elts.len() - 1u;
        } else { self.hi -= 1u; }
        let mut result = self.elts[self.hi].swap_unwrap();
        self.elts[self.hi] = None;
        self.nelts -= 1u;
        result
    }

    fn add_front(&mut self, t: T) {
        let oldlo = self.lo;
        if self.lo == 0u {
            self.lo = self.elts.len() - 1u;
        } else { self.lo -= 1u; }
        if self.lo == self.hi {
            self.elts = grow(self.nelts, oldlo, self.elts);
            self.lo = self.elts.len() - 1u;
            self.hi = self.nelts;
        }
        self.elts[self.lo] = Some(t);
        self.nelts += 1u;
    }

    fn add_back(&mut self, t: T) {
        if self.lo == self.hi && self.nelts != 0u {
            self.elts = grow(self.nelts, self.lo, self.elts);
            self.lo = 0u;
            self.hi = self.nelts;
        }
        self.elts[self.hi] = Some(t);
        self.hi = (self.hi + 1u) % self.elts.len();
        self.nelts += 1u;
    }
}

/// Grow is only called on full elts, so nelts is also len(elts), unlike
/// elsewhere.
fn grow<T>(nelts: uint, lo: uint, elts: &mut [Option<T>]) -> ~[Option<T>] {
    fail_unless!(nelts == elts.len());
    let mut rv = ~[];

    do rv.grow_fn(nelts + 1) |i| {
        let mut element = None;
        element <-> elts[(lo + i) % nelts];
        element
    }

    rv
}

fn get<T>(elts: &r/[Option<T>], i: uint) -> &r/T {
    match elts[i] { Some(ref t) => t, _ => fail!() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Eq;
    use core::kinds::{Durable, Copy};
    use core::prelude::debug;

    #[test]
    fn test_simple() {
        let mut d = Deque::new();
        fail_unless!(d.len() == 0u);
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        fail_unless!(d.len() == 3u);
        d.add_back(137);
        fail_unless!(d.len() == 4u);
        log(debug, d.peek_front());
        fail_unless!(*d.peek_front() == 42);
        log(debug, d.peek_back());
        fail_unless!(*d.peek_back() == 137);
        let mut i: int = d.pop_front();
        log(debug, i);
        fail_unless!(i == 42);
        i = d.pop_back();
        log(debug, i);
        fail_unless!(i == 137);
        i = d.pop_back();
        log(debug, i);
        fail_unless!(i == 137);
        i = d.pop_back();
        log(debug, i);
        fail_unless!(i == 17);
        fail_unless!(d.len() == 0u);
        d.add_back(3);
        fail_unless!(d.len() == 1u);
        d.add_front(2);
        fail_unless!(d.len() == 2u);
        d.add_back(4);
        fail_unless!(d.len() == 3u);
        d.add_front(1);
        fail_unless!(d.len() == 4u);
        log(debug, d.get(0));
        log(debug, d.get(1));
        log(debug, d.get(2));
        log(debug, d.get(3));
        fail_unless!(*d.get(0) == 1);
        fail_unless!(*d.get(1) == 2);
        fail_unless!(*d.get(2) == 3);
        fail_unless!(*d.get(3) == 4);
    }

    #[test]
    fn test_boxes() {
        let a: @int = @5;
        let b: @int = @72;
        let c: @int = @64;
        let d: @int = @175;

        let mut deq = Deque::new();
        fail_unless!(deq.len() == 0);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        fail_unless!(deq.len() == 3);
        deq.add_back(d);
        fail_unless!(deq.len() == 4);
        fail_unless!(*deq.peek_front() == b);
        fail_unless!(*deq.peek_back() == d);
        fail_unless!(deq.pop_front() == b);
        fail_unless!(deq.pop_back() == d);
        fail_unless!(deq.pop_back() == c);
        fail_unless!(deq.pop_back() == a);
        fail_unless!(deq.len() == 0);
        deq.add_back(c);
        fail_unless!(deq.len() == 1);
        deq.add_front(b);
        fail_unless!(deq.len() == 2);
        deq.add_back(d);
        fail_unless!(deq.len() == 3);
        deq.add_front(a);
        fail_unless!(deq.len() == 4);
        fail_unless!(*deq.get(0) == a);
        fail_unless!(*deq.get(1) == b);
        fail_unless!(*deq.get(2) == c);
        fail_unless!(*deq.get(3) == d);
    }

    fn test_parameterized<T:Copy + Eq + Durable>(a: T, b: T, c: T, d: T) {
        let mut deq = Deque::new();
        fail_unless!(deq.len() == 0);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        fail_unless!(deq.len() == 3);
        deq.add_back(d);
        fail_unless!(deq.len() == 4);
        fail_unless!(*deq.peek_front() == b);
        fail_unless!(*deq.peek_back() == d);
        fail_unless!(deq.pop_front() == b);
        fail_unless!(deq.pop_back() == d);
        fail_unless!(deq.pop_back() == c);
        fail_unless!(deq.pop_back() == a);
        fail_unless!(deq.len() == 0);
        deq.add_back(c);
        fail_unless!(deq.len() == 1);
        deq.add_front(b);
        fail_unless!(deq.len() == 2);
        deq.add_back(d);
        fail_unless!(deq.len() == 3);
        deq.add_front(a);
        fail_unless!(deq.len() == 4);
        fail_unless!(*deq.get(0) == a);
        fail_unless!(*deq.get(1) == b);
        fail_unless!(*deq.get(2) == c);
        fail_unless!(*deq.get(3) == d);
    }

    #[deriving_eq]
    enum Taggy { One(int), Two(int, int), Three(int, int, int), }

    #[deriving_eq]
    enum Taggypar<T> {
        Onepar(int), Twopar(int, int), Threepar(int, int, int),
    }

    #[deriving_eq]
    struct RecCy {
        x: int,
        y: int,
        t: Taggy
    }

    #[test]
    fn test_param_int() {
        test_parameterized::<int>(5, 72, 64, 175);
    }

    #[test]
    fn test_param_at_int() {
        test_parameterized::<@int>(@5, @72, @64, @175);
    }

    #[test]
    fn test_param_taggy() {
        test_parameterized::<Taggy>(One(1), Two(1, 2), Three(1, 2, 3),
                                    Two(17, 42));
    }

    #[test]
    fn test_param_taggypar() {
        test_parameterized::<Taggypar<int>>(Onepar::<int>(1),
                                            Twopar::<int>(1, 2),
                                            Threepar::<int>(1, 2, 3),
                                            Twopar::<int>(17, 42));
    }

    #[test]
    fn test_param_reccy() {
        let reccy1 = RecCy { x: 1, y: 2, t: One(1) };
        let reccy2 = RecCy { x: 345, y: 2, t: Two(1, 2) };
        let reccy3 = RecCy { x: 1, y: 777, t: Three(1, 2, 3) };
        let reccy4 = RecCy { x: 19, y: 252, t: Two(17, 42) };
        test_parameterized::<RecCy>(reccy1, reccy2, reccy3, reccy4);
    }
}
