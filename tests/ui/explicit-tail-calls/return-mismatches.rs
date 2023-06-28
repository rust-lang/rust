#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

fn _f0<'a>() -> &'static [u8] {
    become _g0(); //~ error: mismatched types
}

fn _g0() -> &'static [u8; 1] {
    &[0]
}

fn _f1() {
    become _g1(); //~ error: mismatched types
}

fn _g1() -> ! {
    become _g1();
}

fn _f2() -> u32 {
    become _g2(); //~ error: mismatched types
}

fn _g2() -> u16 {
    0
}

fn main() {}
