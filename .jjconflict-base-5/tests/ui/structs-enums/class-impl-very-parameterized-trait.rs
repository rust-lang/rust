//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::cmp;

#[derive(Copy, Clone, Debug)]
enum cat_type { tuxedo, tabby, tortoiseshell }

impl cmp::PartialEq for cat_type {
    fn eq(&self, other: &cat_type) -> bool {
        ((*self) as usize) == ((*other) as usize)
    }
    fn ne(&self, other: &cat_type) -> bool { !(*self).eq(other) }
}

// Very silly -- this just returns the value of the name field
// for any isize value that's less than the meows field

// ok: T should be in scope when resolving the trait ref for map
struct cat<T> {
    // Yes, you can have negative meows
    meows : isize,

    how_hungry : isize,
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
    fn len(&self) -> usize { self.meows as usize }
    fn is_empty(&self) -> bool { self.meows == 0 }
    fn clear(&mut self) {}
    fn contains_key(&self, k: &isize) -> bool { *k <= self.meows }

    fn find(&self, k: &isize) -> Option<&T> {
        if *k <= self.meows {
            Some(&self.name)
        } else {
            None
        }
    }
    fn insert(&mut self, k: isize, _: T) -> bool {
        self.meows += k;
        true
    }

    fn find_mut(&mut self, _k: &isize) -> Option<&mut T> { panic!() }

    fn remove(&mut self, k: &isize) -> bool {
        if self.find(k).is_some() {
            self.meows -= *k; true
        } else {
            false
        }
    }

    fn pop(&mut self, _k: &isize) -> Option<T> { panic!() }

    fn swap(&mut self, _k: isize, _v: T) -> Option<T> { panic!() }
}

impl<T> cat<T> {
    pub fn get(&self, k: &isize) -> &T {
        match self.find(k) {
          Some(v) => { v }
          None    => { panic!("epic fail"); }
        }
    }

    pub fn new(in_x: isize, in_y: isize, in_name: T) -> cat<T> {
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
    for _ in 1_usize..5 { nyan.speak(); }
    assert_eq!(*nyan.find(&1).unwrap(), "nyan".to_string());
    assert_eq!(nyan.find(&10), None);
    let mut spotty: cat<cat_type> = cat::new(2, 57, cat_type::tuxedo);
    for _ in 0_usize..6 { spotty.speak(); }
    assert_eq!(spotty.len(), 8);
    assert!(spotty.contains_key(&2));
    assert_eq!(spotty.get(&3), &cat_type::tuxedo);
}
