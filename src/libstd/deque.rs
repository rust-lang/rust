// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A deque. Untested as of yet. Likely buggy
#[forbid(deprecated_mode)];
#[forbid(non_camel_case_types)];

use option::{Some, None};
use dvec::DVec;
use core::cmp::{Eq};

pub trait Deque<T> {
    fn size() -> uint;
    fn add_front(v: T);
    fn add_back(v: T);
    fn pop_front() -> T;
    fn pop_back() -> T;
    fn peek_front() -> T;
    fn peek_back() -> T;
    fn get(int) -> T;
}

// FIXME (#2343) eventually, a proper datatype plus an exported impl would
// be preferrable.
pub fn create<T: Copy>() -> Deque<T> {
    type Cell<T> = Option<T>;

    let initial_capacity: uint = 32u; // 2^5
     /**
      * Grow is only called on full elts, so nelts is also len(elts), unlike
      * elsewhere.
      */
    fn grow<T: Copy>(nelts: uint, lo: uint, elts: ~[Cell<T>])
      -> ~[Cell<T>] {
        let mut elts = move elts;
        assert (nelts == vec::len(elts));
        let mut rv = ~[];

        let mut i = 0u;
        let nalloc = uint::next_power_of_two(nelts + 1u);
        while i < nalloc {
            if i < nelts {
                rv.push(elts[(lo + i) % nelts]);
            } else { rv.push(None); }
            i += 1u;
        }

        move rv
    }
    fn get<T: Copy>(elts: &DVec<Cell<T>>, i: uint) -> T {
        match (*elts).get_elt(i) { Some(move t) => t, _ => fail }
    }

    type Repr<T> = {mut nelts: uint,
                    mut lo: uint,
                    mut hi: uint,
                    elts: DVec<Cell<T>>};

    impl <T: Copy> Repr<T>: Deque<T> {
        fn size() -> uint { return self.nelts; }
        fn add_front(t: T) {
            let oldlo: uint = self.lo;
            if self.lo == 0u {
                self.lo = self.elts.len() - 1u;
            } else { self.lo -= 1u; }
            if self.lo == self.hi {
                self.elts.swap(|v| grow(self.nelts, oldlo, move v));
                self.lo = self.elts.len() - 1u;
                self.hi = self.nelts;
            }
            self.elts.set_elt(self.lo, Some(t));
            self.nelts += 1u;
        }
        fn add_back(t: T) {
            if self.lo == self.hi && self.nelts != 0u {
                self.elts.swap(|v| grow(self.nelts, self.lo, move v));
                self.lo = 0u;
                self.hi = self.nelts;
            }
            self.elts.set_elt(self.hi, Some(t));
            self.hi = (self.hi + 1u) % self.elts.len();
            self.nelts += 1u;
        }
        /**
         * We actually release (turn to none()) the T we're popping so
         * that we don't keep anyone's refcount up unexpectedly.
         */
        fn pop_front() -> T {
            let t: T = get(&self.elts, self.lo);
            self.elts.set_elt(self.lo, None);
            self.lo = (self.lo + 1u) % self.elts.len();
            self.nelts -= 1u;
            return t;
        }
        fn pop_back() -> T {
            if self.hi == 0u {
                self.hi = self.elts.len() - 1u;
            } else { self.hi -= 1u; }
            let t: T = get(&self.elts, self.hi);
            self.elts.set_elt(self.hi, None);
            self.nelts -= 1u;
            return t;
        }
        fn peek_front() -> T { return get(&self.elts, self.lo); }
        fn peek_back() -> T { return get(&self.elts, self.hi - 1u); }
        fn get(i: int) -> T {
            let idx = (self.lo + (i as uint)) % self.elts.len();
            return get(&self.elts, idx);
        }
    }

    let repr: Repr<T> = {
        mut nelts: 0u,
        mut lo: 0u,
        mut hi: 0u,
        elts:
            dvec::from_vec(
                vec::from_elem(initial_capacity, None))
    };
    (move repr) as Deque::<T>
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_simple() {
        let d: deque::Deque<int> = deque::create::<int>();
        assert (d.size() == 0u);
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        assert (d.size() == 3u);
        d.add_back(137);
        assert (d.size() == 4u);
        log(debug, d.peek_front());
        assert (d.peek_front() == 42);
        log(debug, d.peek_back());
        assert (d.peek_back() == 137);
        let mut i: int = d.pop_front();
        log(debug, i);
        assert (i == 42);
        i = d.pop_back();
        log(debug, i);
        assert (i == 137);
        i = d.pop_back();
        log(debug, i);
        assert (i == 137);
        i = d.pop_back();
        log(debug, i);
        assert (i == 17);
        assert (d.size() == 0u);
        d.add_back(3);
        assert (d.size() == 1u);
        d.add_front(2);
        assert (d.size() == 2u);
        d.add_back(4);
        assert (d.size() == 3u);
        d.add_front(1);
        assert (d.size() == 4u);
        log(debug, d.get(0));
        log(debug, d.get(1));
        log(debug, d.get(2));
        log(debug, d.get(3));
        assert (d.get(0) == 1);
        assert (d.get(1) == 2);
        assert (d.get(2) == 3);
        assert (d.get(3) == 4);
    }

    #[test]
    fn test_boxes() {
        let a: @int = @5;
        let b: @int = @72;
        let c: @int = @64;
        let d: @int = @175;

        let deq: deque::Deque<@int> = deque::create::<@int>();
        assert (deq.size() == 0u);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert (deq.size() == 3u);
        deq.add_back(d);
        assert (deq.size() == 4u);
        assert (deq.peek_front() == b);
        assert (deq.peek_back() == d);
        assert (deq.pop_front() == b);
        assert (deq.pop_back() == d);
        assert (deq.pop_back() == c);
        assert (deq.pop_back() == a);
        assert (deq.size() == 0u);
        deq.add_back(c);
        assert (deq.size() == 1u);
        deq.add_front(b);
        assert (deq.size() == 2u);
        deq.add_back(d);
        assert (deq.size() == 3u);
        deq.add_front(a);
        assert (deq.size() == 4u);
        assert (deq.get(0) == a);
        assert (deq.get(1) == b);
        assert (deq.get(2) == c);
        assert (deq.get(3) == d);
    }

    fn test_parameterized<T: Copy Eq Owned>(a: T, b: T, c: T, d: T) {
        let deq: deque::Deque<T> = deque::create::<T>();
        assert (deq.size() == 0u);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert (deq.size() == 3u);
        deq.add_back(d);
        assert (deq.size() == 4u);
        assert deq.peek_front() == b;
        assert deq.peek_back() == d;
        assert deq.pop_front() == b;
        assert deq.pop_back() == d;
        assert deq.pop_back() == c;
        assert deq.pop_back() == a;
        assert (deq.size() == 0u);
        deq.add_back(c);
        assert (deq.size() == 1u);
        deq.add_front(b);
        assert (deq.size() == 2u);
        deq.add_back(d);
        assert (deq.size() == 3u);
        deq.add_front(a);
        assert (deq.size() == 4u);
        assert deq.get(0) == a;
        assert deq.get(1) == b;
        assert deq.get(2) == c;
        assert deq.get(3) == d;
    }

    enum Taggy { One(int), Two(int, int), Three(int, int, int), }

    enum Taggypar<T> {
        Onepar(int), Twopar(int, int), Threepar(int, int, int),
    }

    type RecCy = {x: int, y: int, t: Taggy};

    impl Taggy : Eq {
        pure fn eq(&self, other: &Taggy) -> bool {
            match (*self) {
              One(a1) => match (*other) {
                One(b1) => return a1 == b1,
                _ => return false
              },
              Two(a1, a2) => match (*other) {
                Two(b1, b2) => return a1 == b1 && a2 == b2,
                _ => return false
              },
              Three(a1, a2, a3) => match (*other) {
                Three(b1, b2, b3) => return a1 == b1 && a2 == b2 && a3 == b3,
                _ => return false
              }
            }
        }
        pure fn ne(&self, other: &Taggy) -> bool { !(*self).eq(other) }
    }

    impl Taggypar<int> : Eq {
        //let eq4: EqFn<Taggypar<int>> = |x,y| taggypareq::<int>(x, y);
        pure fn eq(&self, other: &Taggypar<int>) -> bool {
                  match (*self) {
                    Onepar::<int>(a1) => match (*other) {
                      Onepar::<int>(b1) => return a1 == b1,
                      _ => return false
                    },
                    Twopar::<int>(a1, a2) => match (*other) {
                      Twopar::<int>(b1, b2) => return a1 == b1 && a2 == b2,
                      _ => return false
                    },
                    Threepar::<int>(a1, a2, a3) => match (*other) {
                      Threepar::<int>(b1, b2, b3) => {
                          return a1 == b1 && a2 == b2 && a3 == b3
                      }
                      _ => return false
                    }
                  }
        }
        pure fn ne(&self, other: &Taggypar<int>) -> bool {
            !(*self).eq(other)
        }
    }

    impl RecCy : Eq {
        pure fn eq(&self, other: &RecCy) -> bool {
          return (*self).x == (*other).x && (*self).y == (*other).y &&
                 (*self).t == (*other).t;
        }
        pure fn ne(&self, other: &RecCy) -> bool { !(*self).eq(other) }
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
        let reccy1: RecCy = {x: 1, y: 2, t: One(1)};
        let reccy2: RecCy = {x: 345, y: 2, t: Two(1, 2)};
        let reccy3: RecCy = {x: 1, y: 777, t: Three(1, 2, 3)};
        let reccy4: RecCy = {x: 19, y: 252, t: Two(17, 42)};
        test_parameterized::<RecCy>(reccy1, reccy2, reccy3, reccy4);
    }
}
