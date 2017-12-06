// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ensure borrowck messages are correct outside special case

fn main() {
    let mut void = ();

    let first = &mut void;
    let second = &mut void; //~ ERROR cannot borrow

    loop {
        let mut inner_void = ();

        let inner_first = &mut inner_void;
        let inner_second = &mut inner_void; //~ ERROR cannot borrow
    }
}

