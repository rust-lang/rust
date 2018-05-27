// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--edition 2018
// aux-build:inject-2015-use-root-module-lib.rs

// The macro `inject_me_at_the_root!` generates some code that uses
// `use x::y` to name the global item `x`. In Rust 2018, that should
// be `use crate::x::y`, but we test here that we still accept it,
// as `inject_2015_lib` is in the 2015 edition.

#[macro_use]
extern crate inject_2015_use_root_module_lib;

inject_me_at_the_root!(x, y);

fn main() {
    println!("Hello, world: {}", y());

    // This path comes out as an error, because `x::y` comes from Rust 2018
    print_me!(x::y); //~ ERROR unresolved import `x::y`
}
