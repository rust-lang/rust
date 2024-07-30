#![feature(offset_of_enum)]

use std::mem::offset_of;

enum Alpha {
    One(u8),
    Two(u8),
}

fn main() {
    offset_of!(Alpha::One, 0); //~ ERROR expected type, found variant `Alpha::One`
    offset_of!(Alpha, One); //~ ERROR `One` is an enum variant; expected field at end of `offset_of`
    offset_of!(Alpha, Two.0);
    offset_of!(Alpha, Two.1); //~ ERROR no field named `1` on enum variant `Alpha::Two`
    offset_of!(Alpha, Two.foo); //~ ERROR no field named `foo` on enum variant `Alpha::Two`
    offset_of!(Alpha, NonExistent); //~ ERROR no variant named `NonExistent` found for enum `Alpha`
    offset_of!(Beta, One); //~ ERROR cannot find type `Beta` in this scope
}
