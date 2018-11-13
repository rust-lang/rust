#![feature(const_raw_ptr_deref)]

const FOO: &str = unsafe { &*(1_usize as *const [u8; 0] as *const [u8] as *const str) };
//~^ ERROR it is undefined behaviour to use this value

fn main() {}
