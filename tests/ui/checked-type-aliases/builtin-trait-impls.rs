//! Built-in impls for checked alias types
//! Regression test for <https://github.com/rust-lang/rust/issues/157755> and
//!     <https://github.com/rust-lang/rust/issues/159264>

//@ check-pass
//@ edition: 2024

#![feature(
    async_drop,
    checked_type_aliases,
    coerce_unsized,
    const_param_ty_trait,
    derive_coerce_pointee,
    dispatch_from_dyn,
    reborrow, unsize
)]
#![allow(incomplete_features)]

use std::pin::Pin;
use std::future::AsyncDrop;
use std::ops::{CoerceUnsized, DispatchFromDyn};
use std::marker::{CoercePointee, CoerceShared, ConstParamTy_, Reborrow, Unsize};

#[derive(Clone)]
struct CopyLocal;
type CopyAlias = CopyLocal;

impl Copy for CopyAlias {}

pub struct DropLocal;
pub type DropAlias = DropLocal;

impl Drop for DropAlias {
    fn drop(&mut self) {}
}

pub struct AsyncDropLocal;
pub type AsyncDropAlias = AsyncDropLocal;

impl AsyncDrop for AsyncDropAlias {
    async fn drop(self: Pin<&mut Self>) {}
}

struct CoerceUnsizedLocal<T> {
    foo: T,
}
type CoerceUnsizedAlias<T> = CoerceUnsizedLocal<T>;

impl<T: CoerceUnsized<U>, U> CoerceUnsized<CoerceUnsizedAlias<U>> for CoerceUnsizedAlias<T> {}


struct ReborrowLocal<'a>(&'a u8);
type ReborrowLocalAlias<'a> = ReborrowLocal<'a>;

impl<'a> Reborrow for ReborrowLocalAlias<'a> {}


struct CoerceSharedMut<'a>(&'a mut u8);
type CoerceSharedMutAlias<'a> = CoerceSharedMut<'a>;

#[derive(Clone, Copy)]
struct CoercedSharedLocal<'a>(&'a u8);
type CoercedShareLocalAlias<'a> = CoercedSharedLocal<'a>;

impl<'a> Reborrow for CoerceSharedMutAlias<'a> {}
impl<'a> CoerceShared<CoercedShareLocalAlias<'a>> for CoerceSharedMutAlias<'a> {}

struct DispatchFromDynLocal<T: ?Sized>(*const T);
type DispatchFromDynLocalAlias<T: ?Sized> = DispatchFromDynLocal<T>;

impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<DispatchFromDynLocalAlias<U>>
    for DispatchFromDynLocalAlias<T> {}


type CoercePointeeLocalAlias<T: ?Sized> = *const T;

#[derive(CoercePointee)]
#[repr(transparent)]
struct CoercePointeeLocal<T: ?Sized> {
    foo: CoercePointeeLocalAlias<T>,
}

#[derive(PartialEq, Eq)]
struct ConstParamTyLocal {
    foo: u32,
}

type ConstParamTyLocalAlias = ConstParamTyLocal;
impl ConstParamTy_ for ConstParamTyLocalAlias { }

fn main() {}
