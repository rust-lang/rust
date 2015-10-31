// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(target_vendor = "x")] //~ ERROR `cfg(target_vendor)` is experimental
#[cfg_attr(target_vendor = "x", x)] //~ ERROR `cfg(target_vendor)` is experimental
struct Foo(u64, u64);

#[cfg(not(any(all(target_vendor = "x"))))] //~ ERROR `cfg(target_vendor)` is experimental
fn foo() {}

fn main() {
    cfg!(target_vendor = "x");
    //~^ ERROR `cfg(target_vendor)` is experimental and subject to change
}
