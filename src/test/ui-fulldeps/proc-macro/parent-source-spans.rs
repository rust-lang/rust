// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:parent-source-spans.rs
// ignore-stage1

#![feature(use_extern_macros, decl_macro, proc_macro_non_items)]

extern crate parent_source_spans;

use parent_source_spans::parent_source_spans;

macro one($a:expr, $b:expr) {
    two!($a, $b);
    //~^ ERROR first parent: "hello"
    //~| ERROR second parent: "world"
}

macro two($a:expr, $b:expr) {
    three!($a, $b);
    //~^ ERROR first final: "hello"
    //~| ERROR second final: "world"
    //~| ERROR first final: "yay"
    //~| ERROR second final: "rust"
}

// forwarding tokens directly doesn't create a new source chain
macro three($($tokens:tt)*) {
    four!($($tokens)*);
}

macro four($($tokens:tt)*) {
    parent_source_spans!($($tokens)*);
}

fn main() {
    one!("hello", "world");
    //~^ ERROR first grandparent: "hello"
    //~| ERROR second grandparent: "world"
    //~| ERROR first source: "hello"
    //~| ERROR second source: "world"

    two!("yay", "rust");
    //~^ ERROR first parent: "yay"
    //~| ERROR second parent: "rust"
    //~| ERROR first source: "yay"
    //~| ERROR second source: "rust"

    three!("hip", "hop");
    //~^ ERROR first final: "hip"
    //~| ERROR second final: "hop"
    //~| ERROR first source: "hip"
    //~| ERROR second source: "hop"
}
