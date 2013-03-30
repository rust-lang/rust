// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static C: &'static int = &1000;
static D: int = *C;
struct S(&'static int);
static E: &'static S = &S(C);
static F: int = ***E;

pub fn main() {
    assert!(D == 1000);
    assert!(F == 1000);
}
