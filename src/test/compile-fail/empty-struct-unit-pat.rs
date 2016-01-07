// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Can't use unit struct as enum pattern

#![feature(rustc_attrs)]
// remove prior feature after warning cycle and promoting warnings to errors
#![feature(braced_empty_structs)]

struct Empty1;

enum E {
    Empty2
}

// remove attribute after warning cycle and promoting warnings to errors
#[rustc_error]
fn main() { //~ ERROR: compilation successful
    let e1 = Empty1;
    let e2 = E::Empty2;

    // Rejected by parser as yet
    // match e1 {
    //     Empty1() => () // ERROR `Empty1` does not name a tuple variant or a tuple struct
    // }
    match e1 {
        Empty1(..) => () //~ WARN `Empty1` does not name a tuple variant or a tuple struct
    }
    // Rejected by parser as yet
    // match e2 {
    //     E::Empty2() => () // ERROR `E::Empty2` does not name a tuple variant or a tuple struct
    // }
    match e2 {
        E::Empty2(..) => () //~ WARN `E::Empty2` does not name a tuple variant or a tuple struct
    }
}
