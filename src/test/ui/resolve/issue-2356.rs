// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Groom {
    fn shave(other: usize);
}

pub struct cat {
  whiskers: isize,
}

pub enum MaybeDog {
    Dog,
    NoDog
}

impl MaybeDog {
  fn bark() {
    // If this provides a suggestion, it's a bug as MaybeDog doesn't impl Groom
    shave();
    //~^ ERROR cannot find function `shave`
    //~| NOTE not found in this scope
  }
}

impl Clone for cat {
  fn clone(&self) -> Self {
    clone();
    //~^ ERROR cannot find function `clone`
    loop {}
  }
}
impl Default for cat {
  fn default() -> Self {
    default();
    //~^ ERROR cannot find function `default`
    loop {}
  }
}

impl Groom for cat {
  fn shave(other: usize) {
    whiskers -= other;
    //~^ ERROR cannot find value `whiskers`
    //~| NOTE `self` value is only available in methods with `self` parameter
    shave(4);
    //~^ ERROR cannot find function `shave`
    purr();
    //~^ ERROR cannot find function `purr`
    //~| NOTE not found in this scope
  }
}

impl cat {
    fn static_method() {}

    fn purr_louder() {
        static_method();
        //~^ ERROR cannot find function `static_method`
        //~| NOTE not found in this scope
        purr();
        //~^ ERROR cannot find function `purr`
        //~| NOTE not found in this scope
        purr();
        //~^ ERROR cannot find function `purr`
        //~| NOTE not found in this scope
        purr();
        //~^ ERROR cannot find function `purr`
        //~| NOTE not found in this scope
    }
}

impl cat {
  fn meow() {
    if self.whiskers > 3 {
        //~^ ERROR expected value, found module `self`
        //~| NOTE `self` value is only available in methods with `self` parameter
        println!("MEOW");
    }
  }

  fn purr(&self) {
    grow_older();
    //~^ ERROR cannot find function `grow_older`
    //~| NOTE not found in this scope
    shave();
    //~^ ERROR cannot find function `shave`
    //~| NOTE not found in this scope
  }

  fn burn_whiskers(&mut self) {
    whiskers = 0;
    //~^ ERROR cannot find value `whiskers`
  }

  pub fn grow_older(other:usize) {
    whiskers = 4;
    //~^ ERROR cannot find value `whiskers`
    //~| NOTE `self` value is only available in methods with `self` parameter
    purr_louder();
    //~^ ERROR cannot find function `purr_louder`
    //~| NOTE not found in this scope
  }
}

fn main() {
    self += 1;
    //~^ ERROR expected value, found module `self`
    //~| NOTE `self` value is only available in methods with `self` parameter
}
