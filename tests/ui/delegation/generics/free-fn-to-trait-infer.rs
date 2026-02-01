//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait<T> {
    fn foo<U>(&self, _: U, _: T) {}
}

impl<T> Trait<T> for u8 {}

reuse Trait::<_>::foo::<i32> as generic_arguments1;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
reuse <u8 as Trait<_>>::foo as generic_arguments2;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR mismatched types
reuse <_ as Trait<_>>::foo as generic_arguments3;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn main() {}
