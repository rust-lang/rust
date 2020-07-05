#![feature(const_fn)]
pub fn abc() {}

pub fn bcd() {}

pub fn cde() {}

pub fn def(_: u8) {}

pub fn efg(a: u8, _: u8) -> u8 {
    a
}

pub fn fgh(a: u8, _: u8) -> u8 {
    a
}

pub fn ghi(a: u8, _: u8) -> u8 {
    a
}

pub fn hij() -> u8 {
    0
}

pub const fn ijk() -> u8 {
    0
}
