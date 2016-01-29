// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that unit and tuple enum variants can be accessed through a type alias of the enum

#![feature(associated_consts)]
enum Foo {
    Unit,
    Bar(i32),
    Baz { i: i32 },
}

impl Foo {
    const Unit: () = ();
}

type Alias = Foo;

impl Default for Foo {
    fn default() -> Self {
        Self::Unit
    }
}

fn main() {
    let _t = Alias::Bar(0);
    let _t = Alias::Baz { i: 0 };

    let _t: () = <Foo>::Unit;
    let _t: () = <Alias>::Unit;

    let _t: Foo = Foo::Unit;
    let t: Foo = Alias::Unit;

    match t {
        Alias::Unit => {}
        Alias::Bar(_i) => {}
        Alias::Baz { i: _i } => {}
    }
}
