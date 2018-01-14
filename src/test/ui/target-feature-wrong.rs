// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-arm
// ignore-aarch64
// ignore-wasm
// ignore-emscripten

#![feature(target_feature)]

#[target_feature = "+sse2"]
//~^ WARN: deprecated
#[target_feature(enable = "foo")]
//~^ ERROR: not valid for this target
#[target_feature(bar)]
//~^ ERROR: only accepts sub-keys
#[target_feature(disable = "baz")]
//~^ ERROR: only accepts sub-keys
unsafe fn foo() {}

#[target_feature(enable = "sse2")]
//~^ ERROR: can only be applied to `unsafe` function
fn bar() {}

fn main() {
    unsafe {
        foo();
        bar();
    }
}
