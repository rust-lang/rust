// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test the error message resulting from a cycle in solving `Foo:
// Sized`. The specifics of the message will of course but the main
// thing we want to preserve is that:
//
// 1. the message should appear attached to one of the structs
//    defined in this file;
// 2. it should elaborate the steps that led to the cycle.

struct Baz { q: Option<Foo> }

struct Foo { q: Option<Baz> }
//~^ ERROR recursive type `Foo` has infinite size
//~| type `Foo` is embedded within `core::option::Option<Foo>`...
//~| ...which in turn is embedded within `core::option::Option<Foo>`...
//~| ...which in turn is embedded within `Baz`...
//~| ...which in turn is embedded within `core::option::Option<Baz>`...
//~| ...which in turn is embedded within `Foo`, completing the cycle.

impl Foo { fn bar(&self) {} }

fn main() {}
