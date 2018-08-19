// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Can't use empty braced struct as constant pattern

// aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty1 {}

enum E {
    Empty3 {}
}

fn main() {
    let e1 = Empty1 {};
    let e3 = E::Empty3 {};
    let xe1 = XEmpty1 {};
    let xe3 = XE::XEmpty3 {};

    match e1 {
        Empty1 => () // Not an error, `Empty1` is interpreted as a new binding
    }
    match e3 {
        E::Empty3 => ()
        //~^ ERROR expected unit struct/variant or constant, found struct variant `E::Empty3`
    }
    match xe1 {
        XEmpty1 => () // Not an error, `XEmpty1` is interpreted as a new binding
    }
    match xe3 {
        XE::XEmpty3 => ()
        //~^ ERROR expected unit struct/variant or constant, found struct variant `XE::XEmpty3`
    }
}
