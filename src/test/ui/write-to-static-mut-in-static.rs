#![feature(const_let)]

pub static mut A: u32 = 0;
pub static mut B: () = unsafe { A = 1; };
//~^ ERROR statements in statics are unstable

pub static mut C: u32 = unsafe { C = 1; 0 };
//~^ ERROR statements in statics are unstable

pub static D: u32 = D;

fn main() {}
