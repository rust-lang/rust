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
//~^ NOTE first definition of type or module `Foo`

#[allow(dead_code)]
struct Foo;
//~^ ERROR duplicate definition of type or module `Foo`


#[allow(non_snake_case)]
mod Bar { }
//~^ NOTE first definition of type or module `Bar`

#[allow(dead_code)]
struct Bar(i32);
//~^ ERROR duplicate definition of type or module `Bar`


#[allow(dead_code)]
struct Baz(i32);
//~^ NOTE first definition of type or module

#[allow(non_snake_case)]
mod Baz { }
//~^ ERROR duplicate definition of type or module `Baz`


#[allow(dead_code)]
struct Qux { x: bool }
//~^ NOTE first definition of type or module

#[allow(non_snake_case)]
mod Qux { }
//~^ ERROR duplicate definition of type or module `Qux`


#[allow(dead_code)]
struct Quux;
//~^ NOTE first definition of type or module

#[allow(non_snake_case)]
mod Quux { }
//~^ ERROR duplicate definition of type or module `Quux`


#[allow(dead_code)]
enum Corge { A, B }

#[allow(non_snake_case)]
mod Corge { }
//~^ ERROR duplicate definition of type or module `Corge`

fn main() { }
