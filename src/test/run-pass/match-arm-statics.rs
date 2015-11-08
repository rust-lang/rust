// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -g

struct NewBool(bool);

enum Direction {
    North,
    East,
    South,
    West
}
struct Foo {
    bar: Option<Direction>,
    baz: NewBool
}
enum EnumWithStructVariants {
    Variant1(bool),
    Variant2 {
        dir: Direction
    }
}

const TRUE_TRUE: (bool, bool) = (true, true);
const NONE: Option<Direction> = None;
const EAST: Direction = Direction::East;
const NEW_FALSE: NewBool = NewBool(false);
const STATIC_FOO: Foo = Foo { bar: Some(Direction::South), baz: NEW_FALSE };
const VARIANT2_NORTH: EnumWithStructVariants = EnumWithStructVariants::Variant2 {
    dir: Direction::North };

pub mod glfw {
    #[derive(Copy, Clone)]
    pub struct InputState(usize);

    pub const RELEASE  : InputState = InputState(0);
    pub const PRESS    : InputState = InputState(1);
    pub const REPEAT   : InputState = InputState(2);
}

fn issue_6533() {
    use glfw;

    fn action_to_str(state: glfw::InputState) -> &'static str {
        use glfw::{RELEASE, PRESS, REPEAT};
        match state {
            RELEASE => { "Released" }
            PRESS   => { "Pressed"  }
            REPEAT  => { "Repeated" }
            _       => { "Unknown"  }
        }
    }

    assert_eq!(action_to_str(glfw::RELEASE), "Released");
    assert_eq!(action_to_str(glfw::PRESS), "Pressed");
    assert_eq!(action_to_str(glfw::REPEAT), "Repeated");
}

fn issue_13626() {
    const VAL: [u8; 1] = [0];
    match [1] {
        VAL => unreachable!(),
        _ => ()
    }
}

fn issue_14576() {
    type Foo = (i32, i32);
    const ON: Foo = (1, 1);
    const OFF: Foo = (0, 0);

    match (1, 1) {
        OFF => unreachable!(),
        ON => (),
        _ => unreachable!()
    }

    enum C { D = 3, E = 4 }
    const F : C = C::D;

    assert_eq!(match C::D { F => 1, _ => 2, }, 1);
}

fn issue_13731() {
    enum A { AA(()) }
    const B: A = A::AA(());

    match A::AA(()) {
        B => ()
    }
}

fn issue_15393() {
    #![allow(dead_code)]
    struct Flags {
        bits: usize
    }

    const FOO: Flags = Flags { bits: 0x01 };
    const BAR: Flags = Flags { bits: 0x02 };
    match (Flags { bits: 0x02 }) {
        FOO => unreachable!(),
        BAR => (),
        _ => unreachable!()
    }
}

fn main() {
    assert_eq!(match (true, false) {
        TRUE_TRUE => 1,
        (false, false) => 2,
        (false, true) => 3,
        (true, false) => 4
    }, 4);

    assert_eq!(match Some(Some(Direction::North)) {
        Some(NONE) => 1,
        Some(Some(Direction::North)) => 2,
        Some(Some(EAST)) => 3,
        Some(Some(Direction::South)) => 4,
        Some(Some(Direction::West)) => 5,
        None => 6
    }, 2);

    assert_eq!(match (Foo { bar: Some(Direction::West), baz: NewBool(true) }) {
        Foo { bar: None, baz: NewBool(true) } => 1,
        Foo { bar: NONE, baz: NEW_FALSE } => 2,
        STATIC_FOO => 3,
        Foo { bar: _, baz: NEW_FALSE } => 4,
        Foo { bar: Some(Direction::West), baz: NewBool(true) } => 5,
        Foo { bar: Some(Direction::South), baz: NewBool(true) } => 6,
        Foo { bar: Some(EAST), .. } => 7,
        Foo { bar: Some(Direction::North), baz: NewBool(true) } => 8
    }, 5);

    assert_eq!(match (EnumWithStructVariants::Variant2 { dir: Direction::North }) {
        EnumWithStructVariants::Variant1(true) => 1,
        EnumWithStructVariants::Variant1(false) => 2,
        EnumWithStructVariants::Variant2 { dir: Direction::West } => 3,
        VARIANT2_NORTH => 4,
        EnumWithStructVariants::Variant2 { dir: Direction::South } => 5,
        EnumWithStructVariants::Variant2 { dir: Direction::East } => 6
    }, 4);

    issue_6533();
    issue_13626();
    issue_13731();
    issue_14576();
    issue_15393();
}
