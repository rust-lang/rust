// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:plugin_intrinsic_codegen.rs
// ignore-stage1

#![feature(plugin, intrinsics)]
#![plugin(plugin_intrinsic_codegen)]

extern "rust-intrinsic" {
    /// The plugin expects 5 generic params
    fn generic_count_mismatch<T>();
    //~^ ERROR intrinsic has wrong number of type parameters: found 1, expected 5
}

fn main() {
    unreachable!();
}
