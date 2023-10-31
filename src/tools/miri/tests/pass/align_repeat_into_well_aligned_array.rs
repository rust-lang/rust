//@compile-flags: -Zmiri-symbolic-alignment-check

use std::mem::size_of;

fn main() {
    let mut a = Params::new();
    // The array itself here happens to be quite well-aligned, but not all its elements have that
    // large alignment and we better make sure that is still accepted by Miri.
    a.key_block = [0; BLOCKBYTES];
}

#[repr(C)]
#[derive(Clone)]
#[allow(unused)]
pub struct Params {
    hash_length: u8,
    key_length: u8,
    key_block: [u8; BLOCKBYTES],
    max_leaf_length: u32,
}

pub const OUTBYTES: usize = 8 * size_of::<u64>();
pub const KEYBYTES: usize = 8 * size_of::<u64>();
pub const BLOCKBYTES: usize = 16 * size_of::<u64>();

impl Params {
    pub fn new() -> Self {
        Self {
            hash_length: OUTBYTES as u8,
            key_length: 0,
            key_block: [0; BLOCKBYTES],
            max_leaf_length: 0,
        }
    }
}
