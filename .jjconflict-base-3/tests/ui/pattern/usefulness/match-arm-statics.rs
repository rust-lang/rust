#![allow(dead_code)]
#![deny(unreachable_patterns)]

use self::Direction::{North, East, South, West};

#[derive(PartialEq, Eq)]
struct NewBool(bool);

#[derive(PartialEq, Eq)]
enum Direction {
    North,
    East,
    South,
    West
}

const TRUE_TRUE: (bool, bool) = (true, true);

fn unreachable_1() {
    match (true, false) {
        TRUE_TRUE => (),
        (false, false) => (),
        (false, true) => (),
        (true, false) => (),
        (true, true) => ()
        //~^ ERROR unreachable pattern
    }
}

const NONE: Option<Direction> = None;
const EAST: Direction = East;

fn unreachable_2() {
    match Some(Some(North)) {
        Some(NONE) => (),
        Some(Some(North)) => (),
        Some(Some(EAST)) => (),
        Some(Some(South)) => (),
        Some(Some(West)) => (),
        Some(Some(East)) => (),
        //~^ ERROR unreachable pattern
        None => ()
    }
}

const NEW_FALSE: NewBool = NewBool(false);
struct Foo {
    bar: Option<Direction>,
    baz: NewBool
}

fn unreachable_3() {
    match (Foo { bar: Some(EAST), baz: NewBool(true) }) {
        Foo { bar: None, baz: NewBool(true) } => (),
        Foo { bar: _, baz: NEW_FALSE } => (),
        Foo { bar: Some(West), baz: NewBool(true) } => (),
        Foo { bar: Some(South), .. } => (),
        Foo { bar: Some(EAST), .. } => (),
        Foo { bar: Some(North), baz: NewBool(true) } => (),
        Foo { bar: Some(EAST), baz: NewBool(false) } => ()
        //~^ ERROR unreachable pattern
    }
}

fn main() {
    unreachable_1();
    unreachable_2();
    unreachable_3();
}
