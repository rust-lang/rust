// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

#![allow(dead_code, unused_variables)]

// Issue #21633:  reject duplicate loop labels in function bodies.
//
// Test rejection of lifetimes in *expressions* that shadow loop labels.

fn foo() {
    // Reusing lifetime `'a` in function item is okay.
    fn foo<'a>(x: &'a i8) -> i8 { *x }

    // So is reusing `'a` in struct item
    struct S1<'a> { x: &'a i8 } impl<'a> S1<'a> { fn m(&self) {} }
    // and a method item
    struct S2; impl S2 { fn m<'a>(&self) {} }

    let z = 3_i8;

    'a: loop { //~ NOTE first declared here
        let b = Box::new(|x: &i8| *x) as Box<for <'a> Fn(&'a i8) -> i8>;
        //~^ WARN lifetime name `'a` shadows a label name that is already in scope
        //~| NOTE lifetime 'a already in scope
        assert_eq!((*b)(&z), z);
        break 'a;
    }
}

#[rustc_error]
pub fn main() { //~ ERROR compilation successful
    foo();
}
