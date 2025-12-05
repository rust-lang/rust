//@ only-64bit

#![feature(const_transmute)]

pub const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
//~^ ERROR transmuting from 8-byte type to 16-byte type
