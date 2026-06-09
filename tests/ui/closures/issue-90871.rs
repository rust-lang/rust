#![feature(type_ascription)]

fn main() {
    type_ascribe!(2, n([u8; || 1]))
    //~^ ERROR cannot find type `n` in this scope
    //~| ERROR mismatched types
}
