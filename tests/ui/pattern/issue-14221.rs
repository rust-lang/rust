#![deny(unreachable_patterns)]
#![allow(unused_variables)]
#![allow(non_snake_case)]

pub enum E {
    A,
    B,
}

pub mod b {
    pub fn key(e: crate::E) -> &'static str {
        match e {
            A => "A",
//~^ ERROR pattern binding `A` is named the same as one of the variants of the type `E`
            B => "B", //~ ERROR: unreachable pattern
//~^ ERROR pattern binding `B` is named the same as one of the variants of the type `E`
        }
    }
}

fn main() {}
