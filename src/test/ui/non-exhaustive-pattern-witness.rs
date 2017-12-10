// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(advanced_slice_patterns)]
#![feature(slice_patterns)]

struct Foo {
    first: bool,
    second: Option<[usize; 4]>
}

fn struct_with_a_nested_enum_and_vector() {
    match (Foo { first: true, second: None }) {
//~^ ERROR non-exhaustive patterns: `Foo { first: false, second: Some([_, _, _, _]) }` not covered
//~| NOTE pattern `Foo { first: false, second: Some([_, _, _, _]) }` not covered
        Foo { first: true, second: None } => (),
        Foo { first: true, second: Some(_) } => (),
        Foo { first: false, second: None } => (),
        Foo { first: false, second: Some([1, 2, 3, 4]) } => ()
    }
}

enum Color {
    Red,
    Green,
    CustomRGBA { a: bool, r: u8, g: u8, b: u8 }
}

fn enum_with_single_missing_variant() {
    match Color::Red {
    //~^ ERROR non-exhaustive patterns: `Red` not covered
    //~| NOTE pattern `Red` not covered
        Color::CustomRGBA { .. } => (),
        Color::Green => ()
    }
}

enum Direction {
    North, East, South, West
}

fn enum_with_multiple_missing_variants() {
    match Direction::North {
    //~^ ERROR non-exhaustive patterns: `East`, `South` and `West` not covered
    //~| NOTE patterns `East`, `South` and `West` not covered
        Direction::North => ()
    }
}

enum ExcessiveEnum {
    First, Second, Third, Fourth, Fifth, Sixth, Seventh, Eighth, Ninth, Tenth, Eleventh, Twelfth
}

fn enum_with_excessive_missing_variants() {
    match ExcessiveEnum::First {
    //~^ ERROR `Second`, `Third`, `Fourth` and 8 more not covered
    //~| NOTE patterns `Second`, `Third`, `Fourth` and 8 more not covered

        ExcessiveEnum::First => ()
    }
}

fn enum_struct_variant() {
    match Color::Red {
    //~^ ERROR non-exhaustive patterns: `CustomRGBA { a: true, .. }` not covered
    //~| NOTE pattern `CustomRGBA { a: true, .. }` not covered
        Color::Red => (),
        Color::Green => (),
        Color::CustomRGBA { a: false, r: _, g: _, b: 0 } => (),
        Color::CustomRGBA { a: false, r: _, g: _, b: _ } => ()
    }
}

enum Enum {
    First,
    Second(bool)
}

fn vectors_with_nested_enums() {
    let x: &'static [Enum] = &[Enum::First, Enum::Second(false)];
    match *x {
    //~^ ERROR non-exhaustive patterns: `[Second(true), Second(false)]` not covered
    //~| NOTE pattern `[Second(true), Second(false)]` not covered
        [] => (),
        [_] => (),
        [Enum::First, _] => (),
        [Enum::Second(true), Enum::First] => (),
        [Enum::Second(true), Enum::Second(true)] => (),
        [Enum::Second(false), _] => (),
        [_, _, ref tail.., _] => ()
    }
}

fn missing_nil() {
    match ((), false) {
    //~^ ERROR non-exhaustive patterns: `((), false)` not covered
    //~| NOTE pattern `((), false)` not covered
        ((), true) => ()
    }
}

fn main() {}
