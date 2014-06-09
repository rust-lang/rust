// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::cmp;

#[deriving(Show)]
enum cat_type { tuxedo, tabby, tortoiseshell }

impl cmp::PartialEq for cat_type {
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
    meows : int,

    how_hungry : int,
    name : T,
}

impl<T> cat<T> {
    pub fn speak(&mut self) { self.meow(); }

    pub fn eat(&mut self) -> bool {
        if self.how_hungry > 0 {
            println!("OM NOM NOM");
            self.how_hungry -= 2;
            return true;
        } else {
            println!("Not hungry!");
            return false;
        }
    }
}

impl<T> Collection for cat<T> {
    fn len(&self) -> uint { self.meows as uint }
    fn is_empty(&self) -> bool { self.meows == 0 }
}

impl<T> Mutable for cat<T> {
    fn clear(&mut self) {}
}

impl<T> Map<int, T> for cat<T> {
    fn contains_key(&self, k: &int) -> bool { *k <= self.meows }

    fn find<'a>(&'a self, k: &int) -> Option<&'a T> {
        if *k <= self.meows {
            Some(&self.name)
        } else {
            None
        }
    }
}

impl<T> MutableMap<int, T> for cat<T> {
    fn insert(&mut self, k: int, _: T) -> bool {
        self.meows += k;
        true
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

impl<T> cat<T> {
    pub fn get<'a>(&'a self, k: &int) -> &'a T {
        match self.find(k) {
          Some(v) => { v }
          None    => { fail!("epic fail"); }
        }
    }

    pub fn new(in_x: int, in_y: int, in_name: T) -> cat<T> {
        cat{meows: in_x, how_hungry: in_y, name: in_name }
    }
}

impl<T> cat<T> {
    fn meow(&mut self) {
        self.meows += 1;
        println!("Meow {}", self.meows);
        if self.meows % 5 == 0 {
            self.how_hungry += 1;
        }
    }
}

pub fn main() {
    let mut nyan: cat<String> = cat::new(0, 2, "nyan".to_string());
    for _ in range(1u, 5) { nyan.speak(); }
    assert!(*nyan.find(&1).unwrap() == "nyan".to_string());
    assert_eq!(nyan.find(&10), None);
    let mut spotty: cat<cat_type> = cat::new(2, 57, tuxedo);
    for _ in range(0u, 6) { spotty.speak(); }
    assert_eq!(spotty.len(), 8);
    assert!((spotty.contains_key(&2)));
    assert_eq!(spotty.get(&3), &tuxedo);
}
