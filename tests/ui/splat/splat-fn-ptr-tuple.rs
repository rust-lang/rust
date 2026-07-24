//! Test using `#[splat]` on tuple arguments of pointers to simple functions.
//! Bug #158603 regression test
//@ run-pass
//@ check-run-results

#![expect(incomplete_features)]
#![feature(splat)]

fn tuple_args(#[splat] (a, b): (u32, i8)) {
    println!("tuple_args: {a} {b}");
}

fn splat_non_terminal_arg(#[splat] (a, b): (u32, i8), c: f64) {
    println!("splat_non_terminal_arg: {a} {b} {c}");
}

fn main() {
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = tuple_args as fn(#[splat] (u32, i8));
    fn_ptr(1, 2);
    fn_ptr(1u32, 2i8);

    #[rustfmt::skip]
    let fn_ptr = tuple_args as fn(#[splat] (u32, i8));
    fn_ptr(1, 2);
    fn_ptr(1u32, 2i8);

    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = tuple_args as _;
    fn_ptr(1, 2);
    fn_ptr(1u32, 2i8);

    // Now without explicit `as`
    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8), f64) = splat_non_terminal_arg;
    fn_ptr(1, 2, 3.5);
    fn_ptr(1u32, 2i8, 3.5f64);

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    #[rustfmt::skip]
    let fn_ptr = splat_non_terminal_arg;
    fn_ptr(1, 2, 3.5);
    fn_ptr(1u32, 2i8, 3.5f64);
}
