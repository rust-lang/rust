// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll

// Test various ways to construct a pointer with a longer lifetime
// than the thing it points at and ensure that they result in
// errors. See also regions-free-region-ordering-callee.rs

struct Paramd<'a> { x: &'a usize }

fn call2<'a, 'b>(a: &'a usize, b: &'b usize) {
    let z: Option<&'b &'a usize> = None;//~ ERROR E0623
}

fn call3<'a, 'b>(a: &'a usize, b: &'b usize) {
    let y: Paramd<'a> = Paramd { x: a };
    let z: Option<&'b Paramd<'a>> = None;//~ ERROR E0623
}

fn call4<'a, 'b>(a: &'a usize, b: &'b usize) {
    let z: Option<&'a &'b usize> = None;//~ ERROR E0623
}

fn main() {}
