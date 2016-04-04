// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for a potential corner case in current impl where you have an
// auto trait (Magic1) that depends on a normal trait (Magic2) which
// in turn depends on the auto trait (Magic1). This was incorrectly
// being considered coinductive, but because of the normal trait
// interfering, it should not be.

#![feature(optin_builtin_traits)]

trait Magic1: Magic2 { }
impl Magic1 for .. {}

trait Magic2 { }
impl<T: Magic1> Magic2 for T { }

fn is_magic1<T: Magic1>() { }

#[derive(Debug)]
struct NoClone;

fn main() {
    is_magic1::<NoClone>(); //~ ERROR E0275
}
