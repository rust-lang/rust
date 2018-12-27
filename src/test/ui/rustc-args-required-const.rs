#![feature(rustc_attrs, const_fn)]

#[rustc_args_required_const(0)]
fn foo(_a: i32) {
}

#[rustc_args_required_const(1)]
fn bar(_a: i32, _b: i32) {
}

const A: i32 = 3;

const fn baz() -> i32 {
    3
}

fn main() {
    foo(2);
    foo(2 + 3);
    const BAZ: i32 = baz();
    foo(BAZ);
    let a = 4;
    foo(A);
    foo(a); //~ ERROR: argument 1 is required to be a constant
    bar(a, 3);
    bar(a, a); //~ ERROR: argument 2 is required to be a constant
}
