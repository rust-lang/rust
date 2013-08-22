// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that even when a type paramenter doesn't implement a required
// super-builtin-kind of a trait, if the type parameter is never used,
// the type can implement the trait anyway.

trait Foo : Send { }

struct X<T>(());

impl <T> Foo for X<T> { }

fn main() { }
