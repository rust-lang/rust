// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

fn main() {
    let x: usize() = 1;
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted

    let b: ::std::boxed()::Box<_> = Box::new(1);
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted

    let p = ::std::str::()::from_utf8(b"foo").unwrap();
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted

    let p = ::std::str::from_utf8::()(b"foo").unwrap();
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted

    let o : Box<::std::marker()::Send> = Box::new(1);
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted

    let o : Box<Send + ::std::marker()::Sync> = Box::new(1);
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted
}

fn foo<X:Default>() {
    let d : X() = Default::default();
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| WARN previously accepted
}
