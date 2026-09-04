//! Regression test for <https://github.com/rust-lang/rust/issues/159558>
#![feature(min_generic_const_args)]
struct S<const N: usize>;

fn foo() -> S<core::direct_const_arg!(_)> {
    //~^ ERROR: type annotations needed
    //~| ERROR: the placeholder `_` is not allowed
    todo!()
}
fn main() {}
