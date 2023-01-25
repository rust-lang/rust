// only-x86_64-unknown-linux-gnu
// check-pass

#![feature(const_transmute)]

const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
