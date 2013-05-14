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

use core::container::{Container, Mutable, Map};
use core::old_iter::BaseIter;

enum cat_type { tuxedo, tabby, tortoiseshell }

impl cmp::Eq for cat_type {
    fn eq(&self, other: &cat_type) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    fn ne(&self, other: &cat_type) -> bool { !(*self).eq(other) }
}

// Very silly -- this just returns the value of the name field
// for any int value that's less than the meows field

// ok: T should be in scope when resolving the trait ref for map
struct cat<T> {
    // Yes, you can have negative meows
    priv meows : int,

    how_hungry : int,
    name : T,
}

pub impl<T> cat<T> {
    fn speak(&mut self) { self.meow(); }

    fn eat(&mut self) -> bool {
        if self.how_hungry > 0 {
            error!("OM NOM NOM");
            self.how_hungry -= 2;
            return true;
        } else {
            error!("Not hungry!");
            return false;
        }
    }
}

impl<T> Container for cat<T> {
    fn len(&const self) -> uint { self.meows as uint }
    fn is_empty(&const self) -> bool { self.meows == 0 }
}

impl<T> Mutable for cat<T> {
    fn clear(&mut self) {}
}

impl<T> Map<int, T> for cat<T> {
    fn each<'a>(&'a self, f: &fn(&int, &'a T) -> bool) -> bool {
        let mut n = int::abs(self.meows);
        while n > 0 {
            if !f(&n, &self.name) { return false; }
            n -= 1;
        }
        return true;
    }

    fn contains_key(&self, k: &int) -> bool { *k <= self.meows }

    fn each_key(&self, f: &fn(v: &int) -> bool) -> bool {
        self.each(|k, _| f(k))
    }

    fn each_value<'a>(&'a self, f: &fn(v: &'a T) -> bool) -> bool {
        self.each(|_, v| f(v))
    }

    fn mutate_values(&mut self, _f: &fn(&int, &mut T) -> bool) -> bool {
        fail!("nope")
    }

    fn insert(&mut self, k: int, _: T) -> bool {
        self.meows += k;
        true
    }

    fn find<'a>(&'a self, k: &int) -> Option<&'a T> {
        if *k <= self.meows {
            Some(&self.name)
        } else {
            None
        }
    }

    fn find_mut<'a>(&'a mut self, _k: &int) -> Option<&'a mut T> { fail!() }

    fn remove(&mut self, k: &int) -> bool {
        if self.find(k).is_some() {
            self.meows -= *k; true
        } else {
            false
        }
    }

    fn pop(&mut self, _k: &int) -> Option<T> { fail!() }

    fn swap(&mut self, _k: int, _v: T) -> Option<T> { fail!() }
}

pub impl<T> cat<T> {
    fn get<'a>(&'a self, k: &int) -> &'a T {
        match self.find(k) {
          Some(v) => { v }
          None    => { fail!("epic fail"); }
        }
    }

    fn new(in_x: int, in_y: int, in_name: T) -> cat<T> {
        cat{meows: in_x, how_hungry: in_y, name: in_name }
    }
}

priv impl<T> cat<T> {
    fn meow(&mut self) {
        self.meows += 1;
        error!("Meow %d", self.meows);
        if self.meows % 5 == 0 {
            self.how_hungry += 1;
        }
    }
}

pub fn main() {
    let mut nyan: cat<~str> = cat::new(0, 2, ~"nyan");
    for uint::range(1, 5) |_| { nyan.speak(); }
    assert!((*nyan.find(&1).unwrap() == ~"nyan"));
    assert!((nyan.find(&10) == None));
    let mut spotty: cat<cat_type> = cat::new(2, 57, tuxedo);
    for uint::range(0, 6) |_| { spotty.speak(); }
    assert!((spotty.len() == 8));
    assert!((spotty.contains_key(&2)));
    assert!((spotty.get(&3) == &tuxedo));
}
