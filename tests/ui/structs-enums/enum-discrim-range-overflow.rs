//@ run-pass
#![allow(overflowing_literals)]


pub enum E64 {
    H64 = 0x7FFF_FFFF_FFFF_FFFF,
    L64 = 0x8000_0000_0000_0000
}
pub enum E32 {
    H32 = 0x7FFF_FFFF,
    L32 = 0x8000_0000
}

pub fn f(e64: E64, e32: E32) -> (bool,bool) {
    (match e64 {
        E64::H64 => true,
        E64::L64 => false
    },
     match e32 {
        E32::H32 => true,
        E32::L32 => false
    })
}

pub fn main() { }
