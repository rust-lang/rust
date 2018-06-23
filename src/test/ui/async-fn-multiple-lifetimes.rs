// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --edition=2018

#![feature(arbitrary_self_types, async_await, await_macro, futures_api, pin)]

use std::ops::Add;

async fn multiple_named_lifetimes<'a, 'b>(_: &'a u8, _: &'b u8) {}
//~^ ERROR multiple different lifetimes used in arguments of `async fn`

async fn multiple_hrtb_and_single_named_lifetime_ok<'c>(
    _: impl for<'a> Add<&'a u8>,
    _: impl for<'b> Add<&'b u8>,
    _: &'c u8,
) {}

async fn multiple_elided_lifetimes(_: &u8, _: &u8) {}
//~^ ERROR multiple elided lifetimes used
//~^^ ERROR missing lifetime specifier

fn main() {}
