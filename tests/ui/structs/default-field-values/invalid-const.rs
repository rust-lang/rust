#![feature(default_field_values)]

pub struct Bat {
    pub bax: u8 = panic!("asdf"), //~ ERROR evaluation panicked: asdf
    pub bat: u8 = 130 + 130, //~ ERROR attempt to compute `130_u8 + 130_u8`, which would overflow
}

pub struct Baz<const C: u8> {
    pub bax: u8 = 130 + C, // ok
    pub bat: u8 = 130 + 130, // ok
    pub bay: u8 = 1, // ok
}

fn main() {}
