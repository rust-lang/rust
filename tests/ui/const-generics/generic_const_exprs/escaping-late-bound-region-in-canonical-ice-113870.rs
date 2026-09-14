//! Regression test for https://github.com/rust-lang/rust/issues/113870

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

const fn allow<'b, 'b>() -> usize
//~^ ERROR the name `'b` is already used for a generic parameter in this item's generic parameters
where
    for<'b> [u8; foo::<'a, 'b>()]: Sized,
    //~^ ERROR lifetime name `'b` shadows a lifetime name that is already in scope
    //~| ERROR use of undeclared lifetime name `'a`
    //~| ERROR cannot capture late-bound lifetime in constant
{
    4
}

const fn foo<'a, 'b>() -> usize
where
    &'a (): Sized,
    &'b (): Sized,
{
    4
}
//~^ ERROR `main` function not found in crate
