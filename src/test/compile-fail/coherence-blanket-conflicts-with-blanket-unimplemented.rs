// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Show;
use std::default::Default;

// Test that two blanket impls conflict (at least without negative
// bounds).  After all, some other crate could implement Even or Odd
// for the same type (though this crate doesn't implement them at all).

trait MyTrait {
    fn get(&self) -> usize;
}

trait Even : ::std::marker::MarkerTrait { }

trait Odd : ::std::marker::MarkerTrait { }

impl<T:Even> MyTrait for T { //~ ERROR E0119
    fn get(&self) -> usize { 0 }
}

impl<T:Odd> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

fn main() { }
