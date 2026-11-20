// Regression test for https://github.com/rust-lang/rust/issues/151186

#![feature(gca_min_const_items, gca_macroless_args)]
#![allow(incomplete_features)]

trait Maybe<T> {}

trait MyTrait<const F: fn() -> ()> {}
//~^ ERROR using function pointers as const generic parameters is forbidden

fn foo<'a>(x: &'a ()) -> &'a () { x }

impl<T> Maybe<T> for T where T: MyTrait<{ foo }> {}
//~^ ERROR function items cannot be used as const args

fn main() {}
