// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

// compile-flags: -O

fn foo(_: &'static [&'static str]) {}
fn bar(_: &'static [&'static str; 3]) {}
fn baz_i32(_: &'static i32) {}
fn baz_u32(_: &'static u32) {}

fn main() {
    foo(&["a", "b", "c"]);
    bar(&["d", "e", "f"]);

    // make sure that these do not cause trouble despite overflowing
    baz_u32(&(0-1));
    baz_i32(&-std::i32::MIN);
}
