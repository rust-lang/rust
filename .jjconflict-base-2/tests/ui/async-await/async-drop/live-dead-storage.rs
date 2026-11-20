// ex-ice: #140429
//@ compile-flags: -Zlint-mir --crate-type lib
//@ edition:2024
//@ check-pass

#![feature(async_drop)]
#![allow(incomplete_features)]

async fn a<T>(x: T) {}
