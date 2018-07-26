// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main () {
    struct Foo { a: bool };

    let f = Foo(); //~ ERROR E0423
}

fn bar() {
    struct S { x: i32, y: i32 }
    #[derive(PartialEq)]
    struct T {}

    if let S { x: _x, y: 2 } = S { x: 1, y: 2 } { println!("Ok"); }
    //~^ ERROR E0423
    //~|  expected type, found `1`
    if T {} == T {} { println!("Ok"); }
    //~^ ERROR E0423
    //~| ERROR expected expression, found `==`
}

fn foo() {
    for _ in std::ops::Range { start: 0, end: 10 } {}
    //~^ ERROR E0423
    //~| ERROR expected type, found `0`
}
