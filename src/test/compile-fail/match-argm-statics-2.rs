// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::Direction::{North, East, South, West};

struct NewBool(bool);

enum Direction {
    North,
    East,
    South,
    West
}

const TRUE_TRUE: (bool, bool) = (true, true);

fn nonexhaustive_1() {
    match (true, false) {
    //~^ ERROR non-exhaustive patterns: `(true, false)` not covered
        TRUE_TRUE => (),
        (false, false) => (),
        (false, true) => ()
    }
}

const NONE: Option<Direction> = None;
const EAST: Direction = East;

fn nonexhaustive_2() {
    match Some(Some(North)) {
    //~^ ERROR non-exhaustive patterns: `Some(Some(West))` not covered
        Some(NONE) => (),
        Some(Some(North)) => (),
        Some(Some(EAST)) => (),
        Some(Some(South)) => (),
        None => ()
    }
}

const NEW_FALSE: NewBool = NewBool(false);
struct Foo {
    bar: Option<Direction>,
    baz: NewBool
}

const STATIC_FOO: Foo = Foo { bar: None, baz: NEW_FALSE };

fn nonexhaustive_3() {
    match (Foo { bar: Some(North), baz: NewBool(true) }) {
    //~^ ERROR non-exhaustive patterns: `Foo { bar: Some(North), baz: NewBool(true) }`
        Foo { bar: None, baz: NewBool(true) } => (),
        Foo { bar: _, baz: NEW_FALSE } => (),
        Foo { bar: Some(West), baz: NewBool(true) } => (),
        Foo { bar: Some(South), .. } => (),
        Foo { bar: Some(EAST), .. } => ()
    }
}

fn main() {
    nonexhaustive_1();
    nonexhaustive_2();
    nonexhaustive_3();
}

