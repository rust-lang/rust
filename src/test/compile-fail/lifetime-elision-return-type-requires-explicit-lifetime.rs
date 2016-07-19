// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Lifetime annotation needed because we have no arguments.
fn f() -> &isize {    //~ ERROR missing lifetime specifier
//~^ HELP there is no value for it to be borrowed from
//~| HELP consider giving it a 'static lifetime
    panic!()
}

// Lifetime annotation needed because we have two by-reference parameters.
fn g(_x: &isize, _y: &isize) -> &isize {    //~ ERROR missing lifetime specifier
//~^ HELP the signature does not say whether it is borrowed from `_x` or `_y`
    panic!()
}

struct Foo<'a> {
    x: &'a isize,
}

// Lifetime annotation needed because we have two lifetimes: one as a parameter
// and one on the reference.
fn h(_x: &Foo) -> &isize { //~ ERROR missing lifetime specifier
//~^ HELP the signature does not say which one of `_x`'s 2 elided lifetimes it is borrowed from
    panic!()
}

fn i(_x: isize) -> &isize { //~ ERROR missing lifetime specifier
//~^ HELP this function's return type contains a borrowed value
//~| HELP consider giving it an explicit bounded or 'static lifetime
    panic!()
}

fn main() {}
