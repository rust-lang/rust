// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Constants (static variables) can be used to match in patterns, but mutable
// statics cannot. This ensures that there's some form of error if this is
// attempted.

static mut a: int = 3;

fn main() {
    // If they can't be matched against, then it's possible to capture the same
    // name as a variable, hence this should be an unreachable pattern situation
    // instead of spitting out a custom error about some identifier collisions
    // (we should allow shadowing)
    match 4i {
        a => {} //~ ERROR static variables cannot be referenced in a pattern
        _ => {}
    }
}

struct NewBool(bool);
enum Direction {
    North,
    East,
    South,
    West
}
const NEW_FALSE: NewBool = NewBool(false);
struct Foo {
    bar: Option<Direction>,
    baz: NewBool
}

static mut STATIC_MUT_FOO: Foo = Foo { bar: Some(West), baz: NEW_FALSE };

fn mutable_statics() {
    match (Foo { bar: Some(North), baz: NewBool(true) }) {
        Foo { bar: None, baz: NewBool(true) } => (),
        STATIC_MUT_FOO => (),
        //~^ ERROR static variables cannot be referenced in a pattern
        Foo { bar: Some(South), .. } => (),
        Foo { bar: Some(EAST), .. } => (),
        Foo { bar: Some(North), baz: NewBool(true) } => (),
        Foo { bar: Some(EAST), baz: NewBool(false) } => ()
    }
}
