#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn foo<T>() where for<const N: u8 = { T::<0>::A as u8 }> T: Default {}
//~^ ERROR const arguments are not allowed on type parameter `T`
//~| ERROR no associated item named `A` found for type parameter `T` in the current scope

fn main() {}
