//! This test verifies that a direct non-primitive cast from an enum to an integer type
//! is correctly disallowed, even when a `From` implementation exists for that enum.

//@ run-rustfix

#![allow(dead_code, unused_variables)]

enum NonNullary {
    Nullary,
    Other(isize),
}

impl From<NonNullary> for isize {
    fn from(val: NonNullary) -> isize {
        match val {
            NonNullary::Nullary => 0,
            NonNullary::Other(i) => i,
        }
    }
}

fn main() {
    let v = NonNullary::Nullary;
    let val = v as isize;
    //~^ ERROR non-primitive cast: `NonNullary` as `isize` [E0605]
    //~| HELP consider using the `From` trait instead
}
