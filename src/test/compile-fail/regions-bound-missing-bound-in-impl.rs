// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

struct Inv<'a> { // invariant w/r/t 'a
    x: &'a mut &'a isize
}

pub trait Foo<'a, 't> {
    fn no_bound<'b>(self, b: Inv<'b>);
    fn has_bound<'b:'a>(self, b: Inv<'b>);
    fn wrong_bound1<'b,'c,'d:'a+'b>(self, b: Inv<'b>, c: Inv<'c>, d: Inv<'d>);
    fn okay_bound<'b,'c,'d:'a+'b+'c>(self, b: Inv<'b>, c: Inv<'c>, d: Inv<'d>);
    fn another_bound<'x: 'a>(self, x: Inv<'x>, y: Inv<'t>);
}

impl<'a, 't> Foo<'a, 't> for &'a isize {
    fn no_bound<'b:'a>(self, b: Inv<'b>) {
        //~^ ERROR lifetime parameters or bounds on method `no_bound` do not match
    }

    fn has_bound<'b>(self, b: Inv<'b>) {
        //~^ ERROR lifetime parameters or bounds on method `has_bound` do not match
    }

    fn wrong_bound1<'b,'c,'d:'a+'c>(self, b: Inv<'b>, c: Inv<'c>, d: Inv<'d>) {
        //~^ ERROR method `wrong_bound1` has an incompatible type for trait
        //
        // Note: This is a terrible error message. It is caused
        // because, in the trait, 'b is early bound, and in the impl,
        // 'c is early bound, so -- after substitution -- the
        // lifetimes themselves look isomorphic.  We fail because the
        // lifetimes that appear in the types are in the wrong
        // order. This should really be fixed by keeping more
        // information about the lifetime declarations in the trait so
        // that we can compare better to the impl, even in cross-crate
        // cases.
    }

    fn okay_bound<'b,'c,'e:'b+'c>(self, b: Inv<'b>, c: Inv<'c>, e: Inv<'e>) {
    }

    fn another_bound<'x: 't>(self, x: Inv<'x>, y: Inv<'t>) {}
}

fn main() { }
