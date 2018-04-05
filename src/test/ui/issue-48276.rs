// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #48276 - ICE when self type does not match what is
// required by a trait and regions are involved.

trait MyFrom<A> {
    fn from(a: A) -> Self;
}

struct A;

impl<'a, 'b> MyFrom<A> for &'a str {
    fn from(self: &'a Self) -> &'b str {
        //~^ ERROR: method `from` has a `&self` declaration in the impl, but not in the trait
        "asdf"
    }
}

struct B;

impl From<A> for B {
    fn from(&self) -> B {
        //~^ ERROR: method `from` has a `&self` declaration in the impl, but not in the trait
        B
    }
}

impl From<A> for &'static str {
    fn from(&self) -> &'static str {
        //~^ ERROR: method `from` has a `&self` declaration in the impl, but not in the trait
        ""
    }
}

fn main(){}
