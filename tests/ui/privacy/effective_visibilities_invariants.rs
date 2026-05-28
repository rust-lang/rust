// Invariant checking doesn't ICE in some cases with errors (issue #104249).

#![feature(staged_api)] //~ ERROR module has missing stability attribute

pub mod m {} //~ ERROR module has missing stability attribute

pub mod m { //~ ERROR the name `m` is defined multiple times
    mod inner {}
    type Inner = u8;
}

fn main() {}
