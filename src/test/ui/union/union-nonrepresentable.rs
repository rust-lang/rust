#![feature(untagged_unions)]

union U { //~ ERROR recursive type `U` has infinite size
    a: u8,
    b: U,
}

fn main() {}
