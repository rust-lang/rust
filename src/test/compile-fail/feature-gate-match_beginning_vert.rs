// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(dead_code)]
enum Foo {
    A,
    B,
    C,
    D,
    E,
}
use Foo::*;

fn main() {
    let x = Foo::A;
    match x {
        | A => println!("A"),
        //~^ ERROR: Use of a '|' at the beginning of a match arm is experimental (see issue #44101)
        | B | C => println!("BC!"),
        //~^ ERROR: Use of a '|' at the beginning of a match arm is experimental (see issue #44101)
        | _ => {},
        //~^ ERROR: Use of a '|' at the beginning of a match arm is experimental (see issue #44101)
    };
    match x {
        A | B | C => println!("ABC!"),
        _ => {},
    };
}

