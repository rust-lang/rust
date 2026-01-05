//@ check-pass
//@ only-64bit

#![feature(const_transmute)]

pub const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
