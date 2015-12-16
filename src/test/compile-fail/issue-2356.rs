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
    //~^ ERROR: unresolved name `shave`
  }
}

impl Groom for cat {
  fn shave(other: usize) {
    whiskers -= other;
    //~^ ERROR: unresolved name `whiskers`. Did you mean `self.whiskers`?
    shave(4);
    //~^ ERROR: unresolved name `shave`. Did you mean to call `Groom::shave`?
    purr();
    //~^ ERROR: unresolved name `purr`
  }
}

impl cat {
    fn static_method() {}

    fn purr_louder() {
        static_method();
        //~^ ERROR: unresolved name `static_method`
        purr();
        //~^ ERROR: unresolved name `purr`
        purr();
        //~^ ERROR: unresolved name `purr`
        purr();
        //~^ ERROR: unresolved name `purr`
    }
}

impl cat {
  fn meow() {
    if self.whiskers > 3 {
        //~^ ERROR: `self` is not available in a static method. Maybe a `self` argument is missing?
        println!("MEOW");
    }
  }

  fn purr(&self) {
    grow_older();
    //~^ ERROR: unresolved name `grow_older`
    shave();
    //~^ ERROR: unresolved name `shave`
  }

  fn burn_whiskers(&mut self) {
    whiskers = 0;
    //~^ ERROR: unresolved name `whiskers`. Did you mean `self.whiskers`?
  }

  pub fn grow_older(other:usize) {
    whiskers = 4;
    //~^ ERROR: unresolved name `whiskers`. Did you mean `self.whiskers`?
    purr_louder();
    //~^ ERROR: unresolved name `purr_louder`
  }
}

fn main() {
    self += 1;
    //~^ ERROR: unresolved name `self`
    // it's a bug if this suggests a missing `self` as we're not in a method
}
