// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_field_attributes)]

struct Foo {
    present: (),
}

fn main() {
    let foo = Foo { #[cfg(all())] present: () };
    let _ = Foo { #[cfg(any())] present: () };
    //~^ ERROR missing field `present` in initializer of `Foo`
    let _ = Foo { present: (), #[cfg(any())] absent: () };
    let _ = Foo { present: (), #[cfg(all())] absent: () };
    //~^ ERROR struct `Foo` has no field named `absent`
    let Foo { #[cfg(all())] present: () } = foo;
    let Foo { #[cfg(any())] present: () } = foo;
    //~^ ERROR pattern does not mention field `present`
    let Foo { present: (), #[cfg(any())] absent: () } = foo;
    let Foo { present: (), #[cfg(all())] absent: () } = foo;
    //~^ ERROR struct `Foo` does not have a field named `absent`
}
