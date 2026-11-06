//! Test for <https://github.com/rust-lang/rust/issues/157755>.

//@ check-pass
//@ edition: 2024

#![feature(async_drop, checked_type_aliases, coerce_unsized, unsize)]
#![allow(incomplete_features)]

use std::pin::Pin;
use std::future::AsyncDrop;
use std::ops::CoerceUnsized;
use std::marker::Unsize;

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

fn main() {}
