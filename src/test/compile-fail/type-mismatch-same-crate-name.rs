// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:crate_a1.rs
// aux-build:crate_a2.rs

// This tests the extra note reported when a type error deals with
// seemingly identical types.
// The main use case of this error is when there are two crates
// (generally different versions of the same crate) with the same name
// causing a type mismatch. Here, we simulate that error using block-scoped
// aliased `extern crate` declarations.

fn main() {
    let foo2 = {extern crate crate_a2 as a; a::Foo};
    let bar2 = {extern crate crate_a2 as a; a::bar()};
    {
        extern crate crate_a1 as a;
        a::try_foo(foo2);
        //~^ ERROR mismatched types
        //~| Perhaps two different versions of crate `crate_a1`
        //~| expected struct `main::a::Foo`
        //~| expected type `main::a::Foo`
        //~| found type `main::a::Foo`
        a::try_bar(bar2);
        //~^ ERROR mismatched types
        //~| Perhaps two different versions of crate `crate_a1`
        //~| expected trait `main::a::Bar`
        //~| expected type `Box<main::a::Bar + 'static>`
        //~| found type `Box<main::a::Bar>`
    }
}
