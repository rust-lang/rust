// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that shadowed lifetimes generate an error.

struct Foo<'a>(&'a isize);

impl<'a> Foo<'a> {
    //~^ NOTE  first declared here
    fn shadow_in_method<'a>(&'a self) -> &'a isize {
        //~^ ERROR lifetime name `'a` shadows a lifetime name that is already in scope
        //~| NOTE lifetime 'a already in scope
        self.0
    }

    fn shadow_in_type<'b>(&'b self) -> &'b isize {
        //~^ NOTE  first declared here
        let x: for<'b> fn(&'b isize) = panic!();
        //~^ ERROR lifetime name `'b` shadows a lifetime name that is already in scope
        //~| NOTE lifetime 'b already in scope
        self.0
    }

    fn not_shadow_in_item<'b>(&'b self) {
        struct Bar<'a, 'b>(&'a isize, &'b isize); // not a shadow, separate item
        fn foo<'a, 'b>(x: &'a isize, y: &'b isize) { } // same
    }
}

fn main() {
}
