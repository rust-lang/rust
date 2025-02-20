//@aux-build:proc_macros.rs
#![warn(clippy::unneeded_field_pattern)]
#![allow(dead_code, unused, clippy::single_match)]

#[macro_use]
extern crate proc_macros;

struct Foo {
    a: i32,
    b: i32,
    c: i32,
}

fn main() {
    let f = Foo { a: 0, b: 0, c: 0 };

    match f {
        Foo { a: _, b: 0, .. } => {},
        //~^ unneeded_field_pattern
        Foo { a: _, b: _, c: _ } => {},
        //~^ unneeded_field_pattern
    }
    match f {
        Foo { b: 0, .. } => {}, // should be OK
        Foo { .. } => {},       // and the Force might be with this one
    }
    external! {
        let f = Foo { a: 0, b: 0, c: 0 };
        match f {
            Foo { a: _, b: 0, .. } => {},

            Foo { a: _, b: _, c: _ } => {},
        }
    }
}
