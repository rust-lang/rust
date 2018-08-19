// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// edition:2015

#![feature(raw_identifiers)]
#![allow(async_idents)]

#[macro_export]
macro_rules! produces_async {
    () => (pub fn async() {})
}

#[macro_export]
macro_rules! produces_async_raw {
    () => (pub fn r#async() {})
}

#[macro_export]
macro_rules! consumes_async {
    (async) => (1)
}

#[macro_export]
macro_rules! consumes_async_raw {
    (r#async) => (1)
}

#[macro_export]
macro_rules! passes_ident {
    ($i: ident) => ($i)
}
