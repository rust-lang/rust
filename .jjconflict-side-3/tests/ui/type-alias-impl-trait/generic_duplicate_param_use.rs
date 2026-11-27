#![feature(type_alias_impl_trait)]

//! This test checks various cases where we are using the same
//! generic parameter twice in the parameter list of a TAIT.
//! Within defining scopes that is not legal, because the hidden type
//! is not fully defined then. This could cause us to have a TAIT
//! that doesn't have a hidden type for all possible combinations of generic
//! parameters passed to it.

use std::fmt::Debug;

fn main() {}

// test that unused generic parameters are ok
type TwoTys<T, U> = impl Debug;

type TwoLifetimes<'a, 'b> = impl Debug;

type TwoConsts<const X: usize, const Y: usize> = impl Debug;

#[define_opaque(TwoTys)]
fn one_ty<T: Debug>(t: T) -> TwoTys<T, T> {
    //~^ ERROR non-defining opaque type use in defining scope
    t
}

#[define_opaque(TwoLifetimes)]
fn one_lifetime<'a>(t: &'a u32) -> TwoLifetimes<'a, 'a> {
    t
    //~^ ERROR non-defining opaque type use in defining scope
}

#[define_opaque(TwoConsts)]
fn one_const<const N: usize>(t: *mut [u8; N]) -> TwoConsts<N, N> {
    //~^ ERROR non-defining opaque type use in defining scope
    t
}
