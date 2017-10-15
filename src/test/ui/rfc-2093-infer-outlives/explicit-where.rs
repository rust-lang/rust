// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Type U needs to outlive lifetime 'b.
struct Foo<'b, U> {
    bar: Bar<'b, U> //~ Error the parameter type `U` may not live long enough [E0309]
}

struct Bar<'a, T> where T: 'a {
    x: &'a (),
    y: T,
}

fn main() { }
