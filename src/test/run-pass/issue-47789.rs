#![feature(nll)]

static mut x: &'static u32 = &0;

fn foo() {
    unsafe { x = &1; }
}

fn main() { }
