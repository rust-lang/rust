// check-pass
#![feature(lazy_normalization_consts)]
#![allow(incomplete_features)]
pub struct ArpIPv4<'a> {
    _s: &'a u8
}

impl<'a> ArpIPv4<'a> {
    const LENGTH: usize = 20;

    pub fn to_buffer() -> [u8; Self::LENGTH] {
        unimplemented!()
    }
}

fn main() {}
