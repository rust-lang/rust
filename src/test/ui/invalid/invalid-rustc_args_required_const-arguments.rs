#![feature(rustc_attrs)]

#[rustc_args_required_const(0)] //~ ERROR index exceeds number of arguments
fn foo1() {}

#[rustc_args_required_const(1)] //~ ERROR index exceeds number of arguments
fn foo2(_: u8) {}

#[rustc_args_required_const(a)] //~ ERROR arguments should be non-negative integers
fn foo4() {}

#[rustc_args_required_const(1, a, 2, b)] //~ ERROR arguments should be non-negative integers
fn foo5(_: u8, _: u8, _: u8) {}

#[rustc_args_required_const(0)] //~ ERROR attribute should be applied to a function
struct S;

#[rustc_args_required_const(0usize)] //~ ERROR suffixed literals are not allowed in attributes
fn foo6(_: u8) {}

extern {
    #[rustc_args_required_const(1)] //~ ERROR index exceeds number of arguments
    fn foo7(_: u8);
}

#[rustc_args_required_const] //~ ERROR malformed `rustc_args_required_const` attribute
fn bar1() {}

#[rustc_args_required_const = 1] //~ ERROR malformed `rustc_args_required_const` attribute
fn bar2() {}

fn main() {}
