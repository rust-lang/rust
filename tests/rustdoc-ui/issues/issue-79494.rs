// only-x86_64-unknown-linux-gnu

#![feature(const_transmute)]

const ZST: &[u8] = unsafe { std::mem::transmute(1usize) }; //~ ERROR cannot transmute between types of different sizes, or dependently-sized types
