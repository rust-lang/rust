// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant)]

struct Foo {
    first: bool,
    second: Option<[uint, ..4]>
}

enum Color {
    Red,
    Green,
    CustomRGBA { a: bool, r: u8, g: u8, b: u8 }
}

fn struct_with_a_nested_enum_and_vector() {
    match Foo { first: true, second: None } {
    //~^ ERROR non-exhaustive patterns: `Foo{first: false, second: Some([_, _, _, _])}` not covered
        Foo { first: true, second: None } => (),
        Foo { first: true, second: Some(_) } => (),
        Foo { first: false, second: None } => (),
        Foo { first: false, second: Some([1u, 2u, 3u, 4u]) } => ()
    }
}

fn enum_with_multiple_missing_variants() {
    match Red {
    //~^ ERROR non-exhaustive patterns: `Red` not covered
        CustomRGBA { .. } => ()
    }
}

fn enum_struct_variant() {
    match Red {
    //~^ ERROR non-exhaustive patterns: `CustomRGBA{a: true, r: _, g: _, b: _}` not covered
        Red => (),
        Green => (),
        CustomRGBA { a: false, r: _, g: _, b: 0 } => (),
        CustomRGBA { a: false, r: _, g: _, b: _ } => ()
    }
}

enum Enum {
    First,
    Second(bool)
}

fn vectors_with_nested_enums() {
    let x: &'static [Enum] = [First, Second(false)];
    match x {
    //~^ ERROR non-exhaustive patterns: `[Second(true), Second(false)]` not covered
        [] => (),
        [_] => (),
        [First, _] => (),
        [Second(true), First] => (),
        [Second(true), Second(true)] => (),
        [Second(false), _] => (),
        [_, _, ..tail, _] => ()
    }
}

fn main() {
    struct_with_a_nested_enum_and_vector();
    enum_with_multiple_missing_variants();
    enum_struct_variant();
}