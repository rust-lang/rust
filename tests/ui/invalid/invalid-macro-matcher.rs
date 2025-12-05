#![allow(unused_macros)]

macro_rules! invalid {
    _ => (); //~ ERROR invalid macro matcher
}

fn main() {
}
