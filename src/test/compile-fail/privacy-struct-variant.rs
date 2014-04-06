// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:privacy-struct-variant.rs

#![feature(struct_variant)]

extern crate other = "privacy-struct-variant";

mod a {
    pub enum Foo {
        Bar {
            baz: int
        }
    }

    fn test() {
        let foo = Bar { baz: 42 };

        let Bar { baz: _ } = foo;
        match foo { Bar { baz: _ } => {} }
    }
}

fn main() {
    let foo = a::Bar { baz: 42 };
    //~^ ERROR: field `baz` of variant `Bar` of enum `a::Foo` is private

    let a::Bar { baz: _ } = foo;
    //~^ ERROR: field `baz` of variant `Bar` of enum `a::Foo` is private
    match foo { a::Bar { baz: _ } => {} }
    //~^ ERROR: field `baz` of variant `Bar` of enum `a::Foo` is private
    //
    let foo = other::Bar { baz: 42 };
    //~^ ERROR: field `baz` of variant `Bar` of enum `privacy-struct-variant::Foo` is private

    let other::Bar { baz: _ } = foo;
    //~^ ERROR: field `baz` of variant `Bar` of enum `privacy-struct-variant::Foo` is private
    match foo { other::Bar { baz: _ } => {} }
    //~^ ERROR: field `baz` of variant `Bar` of enum `privacy-struct-variant::Foo` is private
}
