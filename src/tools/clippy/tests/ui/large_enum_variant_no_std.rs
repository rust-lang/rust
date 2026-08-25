#![no_std]
#![warn(clippy::large_enum_variant)]

enum Myenum {
    //~^ ERROR: large size difference between variants
    Small(u8),
    Large([u8; 1024]),
}
