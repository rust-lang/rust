// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// In this test baz isn't resolved when called as foo.baz even though
// it's called from inside foo. This is somewhat surprising and may
// want to change eventually.

mod foo {
    pub fn bar() { foo::baz(); } //~ ERROR failed to resolve. Use of undeclared type or module `foo`

    fn baz() { }
}

fn main() { }
