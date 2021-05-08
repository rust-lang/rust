#![feature(rustc_attrs)]

#[rustc_args_required_const(0)]
fn foo(_imm8: i32) {}

fn bar() {
    foo(*&mut 42) //~ ERROR argument 1 is required to be a constant
}

fn main() {}
