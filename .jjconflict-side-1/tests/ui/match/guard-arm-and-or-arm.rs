//! Regression test for <https://github.com/rust-lang/rust/issues/3895>.
//! Match with guard arm and or pattern used to ICE.
//@ run-pass

#![allow(dead_code)]

pub fn main() {
    enum State { BadChar, BadSyntax }

    match State::BadChar {
        _ if true => State::BadChar,
        State::BadChar | State::BadSyntax => panic!() ,
    };
}
