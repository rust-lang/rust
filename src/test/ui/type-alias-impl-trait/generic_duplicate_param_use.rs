#![feature(type_alias_impl_trait, const_generics)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type TwoTys<T, U> = impl Debug;
type TwoLifetimes<'a, 'b> = impl Debug;
type TwoConsts<const X: usize, const Y: usize> = impl Debug;

fn one_ty<T: Debug>(t: T) -> TwoTys<T, T> {
//~^ ERROR non-defining opaque type use in defining scope
    t
}

fn one_lifetime<'a>(t: &'a u32) -> TwoLifetimes<'a, 'a> {
//~^ ERROR non-defining opaque type use in defining scope
    t
}

fn one_const<const N: usize>(t: *mut [u8; N]) -> TwoConsts<N, N> {
//~^ ERROR non-defining opaque type use in defining scope
    t
}
