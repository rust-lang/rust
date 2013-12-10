// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that merely having lifetime parameters is not
// enough for trans to consider this as non-monomorphic,
// which led to various assertions and failures in turn.

struct S<'a> {
    v: &'a int
}

fn f<'lt>(_s: &'lt S<'lt>) {}

pub fn main() {
    f(& S { v: &42 });
}
