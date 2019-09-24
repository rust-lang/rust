// build-pass (FIXME(62277): could be check-pass?)
#![allow(non_upper_case_globals)]

static mut x: &'static u32 = &0;

fn foo() {
    unsafe { x = &1; }
}

fn main() { }
