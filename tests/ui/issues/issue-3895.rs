//@ run-pass
#![allow(dead_code)]

pub fn main() {
    enum State { BadChar, BadSyntax }

    match State::BadChar {
        _ if true => State::BadChar,
        State::BadChar | State::BadSyntax => panic!() ,
    };
}
