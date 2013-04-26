// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A double-ended queue implemented as a circular buffer

static initial_capacity: uint = 32u; // 2^5

pub struct Deque<T> {
    priv nelts: uint,
    priv lo: uint,
    priv hi: uint,
    priv elts: ~[Option<T>]
}

impl<T> Container for Deque<T> {
    /// Return the number of elements in the deque
    fn len(&const self) -> uint { self.nelts }

    /// Return true if the deque contains no elements
    fn is_empty(&const self) -> bool { self.len() == 0 }
}

impl<T> Mutable for Deque<T> {
    /// Clear the deque, removing all values.
    fn clear(&mut self) {
        for self.elts.each_mut |x| { *x = None }
        self.nelts = 0;
        self.lo = 0;
        self.hi = 0;
    }
}

#[cfg(stage0)]
pub impl<T> Deque<T> {
    /// Create an empty Deque
    fn new() -> Deque<T> {
        Deque{nelts: 0, lo: 0, hi: 0,
              elts: vec::from_fn(initial_capacity, |_| None)}
    }

    /// Return a reference to the first element in the deque
    ///
    /// Fails if the deque is empty
    #[cfg(stage0)]
    fn peek_front(&self) -> &'self T { get(self.elts, self.lo) }

    /// Return a reference to the first element in the deque
    ///
    /// Fails if the deque is empty
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn peek_front<'a>(&'a self) -> &'a T { get(self.elts, self.lo) }

    /// Return a reference to the last element in the deque
    ///
    /// Fails if the deque is empty
    #[cfg(stage0)]
    fn peek_back(&self) -> &'self T { get(self.elts, self.hi - 1u) }

    /// Return a reference to the last element in the deque
    ///
    /// Fails if the deque is empty
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn peek_back<'a>(&'a self) -> &'a T { get(self.elts, self.hi - 1u) }

    /// Retrieve an element in the deque by index
    ///
    /// Fails if there is no element with the given index
    #[cfg(stage0)]
    fn get(&self, i: int) -> &'self T {
        let idx = (self.lo + (i as uint)) % self.elts.len();
        get(self.elts, idx)
    }

    /// Retrieve an element in the deque by index
    ///
    /// Fails if there is no element with the given index
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn get<'a>(&'a self, i: int) -> &'a T {
        let idx = (self.lo + (i as uint)) % self.elts.len();
        get(self.elts, idx)
    }

    /// Iterate over the elements in the deque
    fn each(&self, f: &fn(&T) -> bool) {
        self.eachi(|_i, e| f(e))
    }

    /// Iterate over the elements in the deque by index
    fn eachi(&self, f: &fn(uint, &T) -> bool) {
        for uint::range(0, self.nelts) |i| {
            if !f(i, self.get(i as int)) { return; }
        }
    }

    /// Remove and return the first element in the deque
    ///
    /// Fails if the deque is empty
    fn pop_front(&mut self) -> T {
        let result = self.elts[self.lo].swap_unwrap();
        self.lo = (self.lo + 1u) % self.elts.len();
        self.nelts -= 1u;
        result
    }

    /// Remove and return the last element in the deque
    ///
    /// Fails if the deque is empty
    fn pop_back(&mut self) -> T {
        if self.hi == 0u {
            self.hi = self.elts.len() - 1u;
        } else { self.hi -= 1u; }
        let result = self.elts[self.hi].swap_unwrap();
        self.elts[self.hi] = None;
        self.nelts -= 1u;
        result
    }

    /// Prepend an element to the deque
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

    /// Append an element to the deque
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

#[cfg(stage1)]
#[cfg(stage2)]
#[cfg(stage3)]
pub impl<T> Deque<T> {
    /// Create an empty Deque
    fn new() -> Deque<T> {
        Deque{nelts: 0, lo: 0, hi: 0,
              elts: vec::from_fn(initial_capacity, |_| None)}
    }

    /// Return a reference to the first element in the deque
    ///
    /// Fails if the deque is empty
    fn peek_front<'a>(&'a self) -> &'a T { get(self.elts, self.lo) }

    /// Return a reference to the last element in the deque
    ///
    /// Fails if the deque is empty
    fn peek_back<'a>(&'a self) -> &'a T { get(self.elts, self.hi - 1u) }

    /// Retrieve an element in the deque by index
    ///
    /// Fails if there is no element with the given index
    fn get<'a>(&'a self, i: int) -> &'a T {
        let idx = (self.lo + (i as uint)) % self.elts.len();
        get(self.elts, idx)
    }

    /// Iterate over the elements in the deque
    fn each(&self, f: &fn(&T) -> bool) {
        self.eachi(|_i, e| f(e))
    }

    /// Iterate over the elements in the deque by index
    fn eachi(&self, f: &fn(uint, &T) -> bool) {
        for uint::range(0, self.nelts) |i| {
            if !f(i, self.get(i as int)) { return; }
        }
    }

    /// Remove and return the first element in the deque
    ///
    /// Fails if the deque is empty
    fn pop_front(&mut self) -> T {
        let result = self.elts[self.lo].swap_unwrap();
        self.lo = (self.lo + 1u) % self.elts.len();
        self.nelts -= 1u;
        result
    }

    /// Remove and return the last element in the deque
    ///
    /// Fails if the deque is empty
    fn pop_back(&mut self) -> T {
        if self.hi == 0u {
            self.hi = self.elts.len() - 1u;
        } else { self.hi -= 1u; }
        let result = self.elts[self.hi].swap_unwrap();
        self.elts[self.hi] = None;
        self.nelts -= 1u;
        result
    }

    /// Prepend an element to the deque
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

    /// Append an element to the deque
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
    assert!(nelts == elts.len());
    let mut rv = ~[];

    do rv.grow_fn(nelts + 1) |i| {
        let mut element = None;
        element <-> elts[(lo + i) % nelts];
        element
    }

    rv
}

fn get<'r, T>(elts: &'r [Option<T>], i: uint) -> &'r T {
    match elts[i] { Some(ref t) => t, _ => fail!() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Eq;
    use core::kinds::{Durable, Copy};

    #[test]
    fn test_simple() {
        let mut d = Deque::new();
        assert!(d.len() == 0u);
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        assert!(d.len() == 3u);
        d.add_back(137);
        assert!(d.len() == 4u);
        debug!(d.peek_front());
        assert!(*d.peek_front() == 42);
        debug!(d.peek_back());
        assert!(*d.peek_back() == 137);
        let mut i: int = d.pop_front();
        debug!(i);
        assert!(i == 42);
        i = d.pop_back();
        debug!(i);
        assert!(i == 137);
        i = d.pop_back();
        debug!(i);
        assert!(i == 137);
        i = d.pop_back();
        debug!(i);
        assert!(i == 17);
        assert!(d.len() == 0u);
        d.add_back(3);
        assert!(d.len() == 1u);
        d.add_front(2);
        assert!(d.len() == 2u);
        d.add_back(4);
        assert!(d.len() == 3u);
        d.add_front(1);
        assert!(d.len() == 4u);
        debug!(d.get(0));
        debug!(d.get(1));
        debug!(d.get(2));
        debug!(d.get(3));
        assert!(*d.get(0) == 1);
        assert!(*d.get(1) == 2);
        assert!(*d.get(2) == 3);
        assert!(*d.get(3) == 4);
    }

    #[test]
    fn test_boxes() {
        let a: @int = @5;
        let b: @int = @72;
        let c: @int = @64;
        let d: @int = @175;

        let mut deq = Deque::new();
        assert!(deq.len() == 0);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert!(deq.len() == 3);
        deq.add_back(d);
        assert!(deq.len() == 4);
        assert!(*deq.peek_front() == b);
        assert!(*deq.peek_back() == d);
        assert!(deq.pop_front() == b);
        assert!(deq.pop_back() == d);
        assert!(deq.pop_back() == c);
        assert!(deq.pop_back() == a);
        assert!(deq.len() == 0);
        deq.add_back(c);
        assert!(deq.len() == 1);
        deq.add_front(b);
        assert!(deq.len() == 2);
        deq.add_back(d);
        assert!(deq.len() == 3);
        deq.add_front(a);
        assert!(deq.len() == 4);
        assert!(*deq.get(0) == a);
        assert!(*deq.get(1) == b);
        assert!(*deq.get(2) == c);
        assert!(*deq.get(3) == d);
    }

    #[test]
    fn test_parameterized<T:Copy + Eq + Durable>(a: T, b: T, c: T, d: T) {
        let mut deq = Deque::new();
        assert!(deq.len() == 0);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert!(deq.len() == 3);
        deq.add_back(d);
        assert!(deq.len() == 4);
        assert!(*deq.peek_front() == b);
        assert!(*deq.peek_back() == d);
        assert!(deq.pop_front() == b);
        assert!(deq.pop_back() == d);
        assert!(deq.pop_back() == c);
        assert!(deq.pop_back() == a);
        assert!(deq.len() == 0);
        deq.add_back(c);
        assert!(deq.len() == 1);
        deq.add_front(b);
        assert!(deq.len() == 2);
        deq.add_back(d);
        assert!(deq.len() == 3);
        deq.add_front(a);
        assert!(deq.len() == 4);
        assert!(*deq.get(0) == a);
        assert!(*deq.get(1) == b);
        assert!(*deq.get(2) == c);
        assert!(*deq.get(3) == d);
    }

    #[deriving(Eq)]
    enum Taggy { One(int), Two(int, int), Three(int, int, int), }

    #[deriving(Eq)]
    enum Taggypar<T> {
        Onepar(int), Twopar(int, int), Threepar(int, int, int),
    }

    #[deriving(Eq)]
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

    #[test]
    fn test_eachi() {
        let mut deq = Deque::new();
        deq.add_back(1);
        deq.add_back(2);
        deq.add_back(3);

        for deq.eachi |i, e| {
            assert_eq!(*e, i + 1);
        }

        deq.pop_front();

        for deq.eachi |i, e| {
            assert_eq!(*e, i + 2);
        }

    }
}
