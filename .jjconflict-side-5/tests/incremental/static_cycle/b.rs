//@ revisions:rpass1 rpass2
//@ ignore-backends: gcc

#![cfg_attr(rpass2, warn(dead_code))]

pub static mut BAA: *const i8 = unsafe { &raw const BOO as *const i8 };

pub static mut BOO: *const i8 = unsafe { &raw const BAA as *const i8 };

fn main() {}
