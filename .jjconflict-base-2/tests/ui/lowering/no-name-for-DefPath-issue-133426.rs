//! Test for the crash in #133426, caused by an empty symbol being used for a
//! type name.

#![allow(incomplete_features)]
#![feature(never_patterns)]

fn a(
    _: impl Iterator<
        Item = [(); {
            match *todo!() { ! }; //~ ERROR type `!` cannot be dereferenced
        }],
    >,
) {
}

fn b(_: impl Iterator<Item = { match 0 { ! } }>) {}
//~^ ERROR associated const equality is incomplete
//~| ERROR expected type, found constant

fn main() {}
