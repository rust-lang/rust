// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Can't use empty braced struct as constant or constructor function

// aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty1 {}

enum E {
    Empty3 {}
}

fn main() {
    let e1 = Empty1; //~ ERROR `Empty1` is the name of a struct or struct variant
    let e1 = Empty1(); //~ ERROR `Empty1` is the name of a struct or struct variant
    let e3 = E::Empty3; //~ ERROR `E::Empty3` is the name of a struct or struct variant
    let e3 = E::Empty3(); //~ ERROR `E::Empty3` is the name of a struct or struct variant

    let xe1 = XEmpty1; //~ ERROR `XEmpty1` is the name of a struct or struct variant
    let xe1 = XEmpty1(); //~ ERROR `XEmpty1` is the name of a struct or struct variant
    let xe3 = XE::Empty3; //~ ERROR no associated item named `Empty3` found for type
    let xe3 = XE::Empty3(); //~ ERROR no associated item named `Empty3` found for type
}
