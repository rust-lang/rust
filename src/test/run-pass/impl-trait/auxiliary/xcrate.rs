// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

// NOTE commented out due to issue #45994
//pub fn fourway_add(a: i32) -> impl Fn(i32) -> impl Fn(i32) -> impl Fn(i32) -> i32 {
//    move |b| move |c| move |d| a + b + c + d
//}

fn some_internal_fn() -> u32 {
    1
}

// See #40839
pub fn return_closure_accessing_internal_fn() -> impl Fn() -> u32 {
    || {
        some_internal_fn() + 1
    }
}
