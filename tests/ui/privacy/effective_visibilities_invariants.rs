// Invariant checking doesn't ICE in some cases with errors (issue #104249).

#![feature(staged_api)] //~ ERROR module has missing stability attribute

pub mod a {} //~ ERROR module has missing stability attribute

pub mod b {  //~ ERROR module has missing stability attribute
    mod inner {}
    type Inner = u8;
}

fn main() {}
