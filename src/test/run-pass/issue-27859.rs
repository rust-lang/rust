// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi no std::env
// ignore-wasm32 issue 42629

#[inline(never)]
fn foo(a: f32, b: f32) -> f32 {
    a % b
}

#[inline(never)]
fn bar(a: f32, b: f32) -> f32 {
    ((a as f64) % (b as f64)) as f32
}

fn main() {
    let unknown_float = std::env::args().len();
    println!("{}", foo(4.0, unknown_float as f32));
    println!("{}", foo(5.0, (unknown_float as f32) + 1.0));
    println!("{}", bar(6.0, (unknown_float as f32) + 2.0));
    println!("{}", bar(7.0, (unknown_float as f32) + 3.0));
}
