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
    //~^ NOTE shadowed lifetime `'a` declared here
    fn shadow_in_method<'a>(&'a self) -> &'a isize {
        //~^ WARNING lifetime name `'a` shadows another lifetime name that is already in scope
        //~| NOTE deprecated
        self.0
    }

    fn shadow_in_type<'b>(&'b self) -> &'b isize {
        //~^ NOTE shadowed lifetime `'b` declared here
        let x: for<'b> fn(&'b isize) = panic!();
        //~^ WARNING lifetime name `'b` shadows another lifetime name that is already in scope
        //~| NOTE deprecated
        self.0
    }

    fn not_shadow_in_item<'b>(&'b self) {
        struct Bar<'a, 'b>(&'a isize, &'b isize); // not a shadow, separate item
        fn foo<'a, 'b>(x: &'a isize, y: &'b isize) { } // same
    }
}

fn main() {
    // intentional error that occurs after `resolve_lifetime` runs,
    // just to ensure that this test fails to compile; when shadowed
    // lifetimes become either an error or a proper lint, this will
    // not be needed.
    let x: isize = 3_usize; //~ ERROR mismatched types
}
