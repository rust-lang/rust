// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-20646.rs
// ignore-android

#![feature(associated_types)]

extern crate issue_20646;

// @has issue_20646/trait.Trait.html \
//      '//*[@id="associatedtype.Output"]' \
//      'type Output'
pub trait Trait {
    type Output;
}

// @has issue_20646/fn.fun.html \
//      '//*[@class="rust fn"]' 'where T: Trait<Output=i32>'
pub fn fun<T>(_: T) where T: Trait<Output=i32> {}

pub mod reexport {
    // @has issue_20646/reexport/trait.Trait.html \
    //      '//*[@id="associatedtype.Output"]' \
    //      'type Output'
    // @has issue_20646/reexport/fn.fun.html \
    //      '//*[@class="rust fn"]' 'where T: Trait<Output=i32>'
    pub use issue_20646::{Trait, fun};
}
