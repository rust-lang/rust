// This test is a regression test for a bug where we only checked function calls in no-const
// functions for `rustc_args_required_const` arguments. This meant that even though `bar` needs its
// argument to be const, inside a const fn (callable at runtime), the value for it may come from a
// non-constant (namely an argument to the const fn).

#![feature(rustc_attrs)]
const fn foo(a: i32) {
    bar(a); //~ ERROR argument 1 is required to be a constant
}

#[rustc_args_required_const(0)]
const fn bar(_: i32) {}

fn main() {
    // this function call will pass a runtime-value (number of program arguments) to `foo`, which
    // will in turn forward it to `bar`, which expects a compile-time argument
    foo(std::env::args().count() as i32);
}
