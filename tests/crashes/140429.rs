//@ known-bug: #140429
//@ compile-flags: -Zlint-mir --crate-type lib
//@ edition:2024

#![feature(async_drop)]
async fn a<T>(x: T) {}
