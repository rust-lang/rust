#![feature(rustc_attrs)]

#[rustc_args_required_const(0)]
fn foo(_imm8: i32) {}

fn bar() {
    let imm8 = 3;
    foo(imm8) //~ ERROR argument 1 is required to be a constant
}

fn main() {}
