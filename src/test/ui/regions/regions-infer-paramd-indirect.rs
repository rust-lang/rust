// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Check that we correctly infer that b and c must be region
// parameterized because they reference a which requires a region.

type A<'a> = &'a isize;
type B<'a> = Box<A<'a>>;

struct C<'a> {
    f: Box<B<'a>>
}

trait SetF<'a> {
    fn set_f_ok(&mut self, b: Box<B<'a>>);
    fn set_f_bad(&mut self, b: Box<B>);
}

impl<'a> SetF<'a> for C<'a> {
    fn set_f_ok(&mut self, b: Box<B<'a>>) {
        self.f = b;
    }

    fn set_f_bad(&mut self, b: Box<B>) {
        self.f = b;
        //~^ ERROR mismatched types
        //~| expected type `std::boxed::Box<std::boxed::Box<&'a isize>>`
        //~| found type `std::boxed::Box<std::boxed::Box<&isize>>`
        //~| lifetime mismatch
    }
}

fn main() {}
