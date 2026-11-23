//! Regression test for <https://github.com/rust-lang/rust/issues/155658>.
#![feature(gca_min_const_items, gca_macroless_args)]

struct S<const N: usize>()
where
    S<{ main }>:;
//~^ ERROR function items cannot be used as const args

fn main(_: impl ?Sized) {}
//~^ ERROR `main` function has wrong type
//~| ERROR the size for values of type `impl ?Sized` cannot be known at compilation time
