// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A few contrived examples where lifetime should (or should not) be parsed as an object type.
// Lifetimes parsed as types are still rejected later by semantic checks.

// compile-flags: -Z continue-parse-after-error

// `'static` is a lifetime, `'static +` is a type, `'a` is a type
fn g() where
    'static: 'static,
    'static +: 'static + Copy,
    //~^ ERROR at least one non-builtin trait is required for an object type
{}

fn main() {}
