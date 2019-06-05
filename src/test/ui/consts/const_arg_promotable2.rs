#![feature(rustc_attrs)]
const fn foo(a: i32) {
    bar(a); //~ ERROR argument 1 is required to be a constant
}

#[rustc_args_required_const(0)]
const fn bar(_: i32) {}

fn main() {}
