use std::mem::offset_of;

enum Alpha {
    One(u8),
    Two(u8),
}

fn main() {
    offset_of!(Alpha::One, 0); //~ ERROR expected type, found variant `Alpha::One`
    offset_of!(Alpha, One); //~ ERROR `One` is an enum variant; expected field at end of `offset_of`
                            //~| ERROR using enums in offset_of is experimental
    offset_of!(Alpha, Two.0); //~ ERROR using enums in offset_of is experimental
}
