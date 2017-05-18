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
//~^ NOTE previous definition of the module `Foo` here

#[allow(dead_code)]
struct Foo;
//~^ ERROR the name `Foo` is defined multiple times
//~| NOTE `Foo` redefined here
//~| NOTE `Foo` must be defined only once in the type namespace of this module

#[allow(non_snake_case)]
mod Bar { }
//~^ NOTE previous definition of the module `Bar` here

#[allow(dead_code)]
struct Bar(i32);
//~^ ERROR the name `Bar` is defined multiple times
//~| NOTE `Bar` redefined here
//~| NOTE `Bar` must be defined only once in the type namespace of this module


#[allow(dead_code)]
struct Baz(i32);
//~^ NOTE previous definition of the type `Baz` here

#[allow(non_snake_case)]
mod Baz { }
//~^ ERROR the name `Baz` is defined multiple times
//~| NOTE `Baz` redefined here
//~| NOTE `Baz` must be defined only once in the type namespace of this module


#[allow(dead_code)]
struct Qux { x: bool }
//~^ NOTE previous definition of the type `Qux` here

#[allow(non_snake_case)]
mod Qux { }
//~^ ERROR the name `Qux` is defined multiple times
//~| NOTE `Qux` redefined here
//~| NOTE `Qux` must be defined only once in the type namespace of this module


#[allow(dead_code)]
struct Quux;
//~^ NOTE previous definition of the type `Quux` here

#[allow(non_snake_case)]
mod Quux { }
//~^ ERROR the name `Quux` is defined multiple times
//~| NOTE `Quux` redefined here
//~| NOTE `Quux` must be defined only once in the type namespace of this module


#[allow(dead_code)]
enum Corge { A, B }
//~^ NOTE previous definition of the type `Corge` here

#[allow(non_snake_case)]
mod Corge { }
//~^ ERROR the name `Corge` is defined multiple times
//~| NOTE `Corge` redefined here
//~| NOTE `Corge` must be defined only once in the type namespace of this module

fn main() { }
