//! Regression test for <https://github.com/rust-lang/rust/issues/142529>.
#![feature(gca_min_const_items, gca_macroless_args)]

struct Foo<T, U = [u8; std::mem::size_of::<(u8, T)>]>(T, U);
//~^ ERROR function items cannot be used as const args

struct Bar<T, U = [u8; size_of::<T>]>(T, U);
//~^ ERROR function items cannot be used as const args

fn main() {}
