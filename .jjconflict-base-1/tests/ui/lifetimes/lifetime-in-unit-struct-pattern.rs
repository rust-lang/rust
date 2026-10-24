//! Regression test for <https://github.com/rust-lang/rust/issues/34751>.
//! This used to ICE with
//! `assertion failed: !substs.has_regions_escaping_depth(0)`.
//@ check-pass

#![allow(dead_code)]

use std::marker::PhantomData;

fn f<'a>(PhantomData::<&'a u8>: PhantomData<&'a u8>) {}

fn main() {}
