// compile-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

#![feature(nll)]

static mut x: &'static u32 = &0;

fn foo() {
    unsafe { x = &1; }
}

fn main() { }
