// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `local_inner_macros` has no effect if `feature(use_extern_macros)` is not enabled

// aux-build:local_inner_macros.rs

#[macro_use(public_macro)]
extern crate local_inner_macros;

public_macro!(); //~ ERROR cannot find macro `helper2!` in this scope

fn main() {}
