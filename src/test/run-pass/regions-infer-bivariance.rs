// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a type whose lifetime parameters is never used is
// inferred to be bivariant.

use std::kinds::marker;

struct Bivariant<'a>;

fn use1<'short,'long>(c: Bivariant<'short>,
                      _where:Option<&'short &'long ()>) {
    let _: Bivariant<'long> = c;
}

fn use2<'short,'long>(c: Bivariant<'long>,
                      _where:Option<&'short &'long ()>) {
    let _: Bivariant<'short> = c;
}

pub fn main() {}
