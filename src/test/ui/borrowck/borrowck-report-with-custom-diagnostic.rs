// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]
#![allow(dead_code)]
fn main() { #![rustc_error] // rust-lang/rust#49855
    // Original borrow ends at end of function
    let mut x = 1;
    let y = &mut x;
    //~^ mutable borrow occurs here
    let z = &x; //~ ERROR cannot borrow
    //~^ immutable borrow occurs here
    z.use_ref();
    y.use_mut();
}

fn foo() {
    match true {
        true => {
            // Original borrow ends at end of match arm
            let mut x = 1;
            let y = &x;
            //~^ immutable borrow occurs here
            let z = &mut x; //~ ERROR cannot borrow
            //~^ mutable borrow occurs here
            z.use_mut();
            y.use_ref();
        }
        false => ()
    }
}

fn bar() {
    // Original borrow ends at end of closure
    || {
        let mut x = 1;
        let y = &mut x;
        //~^ first mutable borrow occurs here
        let z = &mut x; //~ ERROR cannot borrow
        //~^ second mutable borrow occurs here
        z.use_mut();
        y.use_mut();
    };
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
