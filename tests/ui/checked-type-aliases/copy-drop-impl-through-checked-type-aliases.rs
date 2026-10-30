//! Test for <https://github.com/rust-lang/rust/issues/157755>.
//! for built-in traits

//@ check-pass
//@ edition: 2024

#![feature(async_drop, checked_type_aliases)]
#![allow(incomplete_features)]

use std::pin::Pin;
use std::future::AsyncDrop;

#[derive(Clone)]
pub struct CopyLocal;

pub type CopyAlias = CopyLocal;
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

fn main() {}
