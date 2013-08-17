// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We can export tags without exporting the variants to create a simple
// sort of ADT.

mod foo {
    pub enum t { t1, }

    pub fn f() -> t { return t1; }
}

pub fn main() { let _v: foo::t = foo::f(); }
