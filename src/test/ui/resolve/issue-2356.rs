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
    //~^ ERROR unresolved function `shave`
    //~| NOTE no resolution found
  }
}

impl Clone for cat {
  fn clone(&self) -> Self {
    clone();
    //~^ ERROR unresolved function `clone`
    //~| NOTE did you mean `self.clone(...)`?
    loop {}
  }
}
impl Default for cat {
  fn default() -> Self {
    default();
    //~^ ERROR unresolved function `default`
    //~| NOTE did you mean `Self::default`?
    loop {}
  }
}

impl Groom for cat {
  fn shave(other: usize) {
    whiskers -= other;
    //~^ ERROR unresolved value `whiskers`
    //~| ERROR unresolved value `whiskers`
    //~| NOTE did you mean `self.whiskers`?
    //~| NOTE `self` value is only available in methods with `self` parameter
    shave(4);
    //~^ ERROR unresolved function `shave`
    //~| NOTE did you mean `Self::shave`?
    purr();
    //~^ ERROR unresolved function `purr`
    //~| NOTE no resolution found
  }
}

impl cat {
    fn static_method() {}

    fn purr_louder() {
        static_method();
        //~^ ERROR unresolved function `static_method`
        //~| NOTE no resolution found
        purr();
        //~^ ERROR unresolved function `purr`
        //~| NOTE no resolution found
        purr();
        //~^ ERROR unresolved function `purr`
        //~| NOTE no resolution found
        purr();
        //~^ ERROR unresolved function `purr`
        //~| NOTE no resolution found
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
    //~^ ERROR unresolved function `grow_older`
    //~| NOTE no resolution found
    shave();
    //~^ ERROR unresolved function `shave`
    //~| NOTE no resolution found
  }

  fn burn_whiskers(&mut self) {
    whiskers = 0;
    //~^ ERROR unresolved value `whiskers`
    //~| NOTE did you mean `self.whiskers`?
  }

  pub fn grow_older(other:usize) {
    whiskers = 4;
    //~^ ERROR unresolved value `whiskers`
    //~| ERROR unresolved value `whiskers`
    //~| NOTE did you mean `self.whiskers`?
    //~| NOTE `self` value is only available in methods with `self` parameter
    purr_louder();
    //~^ ERROR unresolved function `purr_louder`
    //~| NOTE no resolution found
  }
}

fn main() {
    self += 1;
    //~^ ERROR expected value, found module `self`
    //~| NOTE `self` value is only available in methods with `self` parameter
}
