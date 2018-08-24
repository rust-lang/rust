#![deny(improper_ctypes)]
#![allow(dead_code)]

enum Z { }
enum U { A }
enum B { C, D }
enum T { E, F, G }

#[repr(C)]
enum ReprC { A, B, C }

#[repr(u8)]
enum U8 { A, B, C }

#[repr(isize)]
enum Isize { A, B, C }

extern {
   fn zf(x: Z);
   fn uf(x: U); //~ ERROR enum has no representation hint
   fn bf(x: B); //~ ERROR enum has no representation hint
   fn tf(x: T); //~ ERROR enum has no representation hint
   fn reprc(x: ReprC);
   fn u8(x: U8);
   fn isize(x: Isize);
}

pub fn main() { }
