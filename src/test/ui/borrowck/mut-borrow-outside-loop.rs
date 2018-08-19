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
#![feature(rustc_attrs)]
fn main() { #![rustc_error] // rust-lang/rust#49855
    let mut void = ();

    let first = &mut void;
    let second = &mut void; //~ ERROR cannot borrow
    first.use_mut();
    second.use_mut();

    loop {
        let mut inner_void = ();

        let inner_first = &mut inner_void;
        let inner_second = &mut inner_void; //~ ERROR cannot borrow
        inner_second.use_mut();
        inner_first.use_mut();
    }
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
