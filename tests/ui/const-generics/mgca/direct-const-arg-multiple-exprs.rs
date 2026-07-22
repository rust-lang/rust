#![feature(min_generic_const_args)]
struct S<const N: usize>;
fn foo<const N: usize, const M: usize>(_: S<core::direct_const_arg!(N, M)>) {}
//~^ ERROR direct_const_arg! takes 1 argument
fn main() {}
