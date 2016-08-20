// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Also works as a test for #14564

#[allow(non_snake_case)]
mod Foo { }
//~^ NOTE previous definition of `Foo` here

#[allow(dead_code)]
struct Foo;
//~^ ERROR a module named `Foo` has already been defined in this module
//~| NOTE already defined

#[allow(non_snake_case)]
mod Bar { }
//~^ NOTE previous definition of `Bar` here

#[allow(dead_code)]
struct Bar(i32);
//~^ ERROR a module named `Bar` has already been defined
//~| NOTE already defined


#[allow(dead_code)]
struct Baz(i32);
//~^ NOTE previous definition

#[allow(non_snake_case)]
mod Baz { }
//~^ ERROR a type named `Baz` has already been defined
//~| NOTE already defined


#[allow(dead_code)]
struct Qux { x: bool }
//~^ NOTE previous definition

#[allow(non_snake_case)]
mod Qux { }
//~^ ERROR a type named `Qux` has already been defined
//~| NOTE already defined


#[allow(dead_code)]
struct Quux;
//~^ NOTE previous definition

#[allow(non_snake_case)]
mod Quux { }
//~^ ERROR a type named `Quux` has already been defined
//~| NOTE already defined


#[allow(dead_code)]
enum Corge { A, B }
//~^ NOTE previous definition

#[allow(non_snake_case)]
mod Corge { }
//~^ ERROR a type named `Corge` has already been defined
//~| NOTE already defined

fn main() { }
