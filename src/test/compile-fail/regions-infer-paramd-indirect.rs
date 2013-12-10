// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// Check that we correctly infer that b and c must be region
// parameterized because they reference a which requires a region.

type a<'a> = &'a int;
type b<'a> = @a<'a>;

struct c<'a> {
    f: @b<'a>
}

trait set_f<'a> {
    fn set_f_ok(&self, b: @b<'a>);
    fn set_f_bad(&self, b: @b);
}

impl<'a> set_f<'a> for c<'a> {
    fn set_f_ok(&self, b: @b<'a>) {
        self.f = b;
    }

    fn set_f_bad(&self, b: @b) {
        self.f = b; //~ ERROR mismatched types: expected `@@&'a int` but found `@@&int`
        //~^ ERROR cannot infer an appropriate lifetime
    }
}

fn main() {}
