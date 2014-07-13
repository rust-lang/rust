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

static TRUE_TRUE: (bool, bool) = (true, true);
static NONE: Option<Direction> = None;
static EAST: Direction = East;
static NEW_FALSE: NewBool = NewBool(false);
static STATIC_FOO: Foo = Foo { bar: Some(South), baz: NEW_FALSE };
static VARIANT2_NORTH: EnumWithStructVariants = Variant2 { dir: North };

pub mod glfw {
    pub struct InputState(uint);

    pub static RELEASE  : InputState = InputState(0);
    pub static PRESS    : InputState = InputState(1);
    pub static REPEAT   : InputState = InputState(2);
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
    static VAL: [u8, ..1] = [0];
    match [1] {
        VAL => unreachable!(),
        _ => ()
    }
}

fn issue_14576() {
    type Foo = (i32, i32);
    static ON: Foo = (1, 1);
    static OFF: Foo = (0, 0);

    match (1, 1) {
        OFF => unreachable!(),
        ON => (),
        _ => unreachable!()
    }

    enum C { D = 3, E = 4 }
    static F : C = D;

    assert_eq!(match D { F => 1i, _ => 2, }, 1);
}

fn issue_13731() {
    enum A { A(()) }
    static B: A = A(());

    match A(()) {
        B => ()
    }
}

fn issue_15393() {
    #![allow(dead_code)]
    struct Flags {
        bits: uint
    }

    static FOO: Flags = Flags { bits: 0x01 };
    static BAR: Flags = Flags { bits: 0x02 };
    match (Flags { bits: 0x02 }) {
        FOO => unreachable!(),
        BAR => (),
        _ => unreachable!()
    }
}

fn main() {
    assert_eq!(match (true, false) {
        TRUE_TRUE => 1i,
        (false, false) => 2,
        (false, true) => 3,
        (true, false) => 4
    }, 4);

    assert_eq!(match Some(Some(North)) {
        Some(NONE) => 1i,
        Some(Some(North)) => 2,
        Some(Some(EAST)) => 3,
        Some(Some(South)) => 4,
        Some(Some(West)) => 5,
        None => 6
    }, 2);

    assert_eq!(match (Foo { bar: Some(West), baz: NewBool(true) }) {
        Foo { bar: None, baz: NewBool(true) } => 1i,
        Foo { bar: NONE, baz: NEW_FALSE } => 2,
        STATIC_FOO => 3,
        Foo { bar: _, baz: NEW_FALSE } => 4,
        Foo { bar: Some(West), baz: NewBool(true) } => 5,
        Foo { bar: Some(South), baz: NewBool(true) } => 6,
        Foo { bar: Some(EAST), .. } => 7,
        Foo { bar: Some(North), baz: NewBool(true) } => 8
    }, 5);

    assert_eq!(match (Variant2 { dir: North }) {
        Variant1(true) => 1i,
        Variant1(false) => 2,
        Variant2 { dir: West } => 3,
        VARIANT2_NORTH => 4,
        Variant2 { dir: South } => 5,
        Variant2 { dir: East } => 6
    }, 4);

    issue_6533();
    issue_13626();
    issue_13731();
    issue_14576();
    issue_15393();
}
