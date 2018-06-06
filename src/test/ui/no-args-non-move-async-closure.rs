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

fn main() {
    let _ = async |x: u8| {};
    //~^ ERROR `async` non-`move` closures with arguments are not currently supported
}
