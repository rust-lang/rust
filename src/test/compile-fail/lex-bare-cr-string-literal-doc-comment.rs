// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-cr

/// doc comment with bare CR: ''
pub fn foo() {}
//~^^ ERROR: bare CR not allowed in doc-comment

/** block doc comment with bare CR: '' */
pub fn bar() {}
//~^^ ERROR: bare CR not allowed in block doc-comment

fn main() {
    // the following string literal has a bare CR in it
    let _s = "foobar"; //~ ERROR: bare CR not allowed in string

    // the following string literal has a bare CR in it
    let _s = r"barfoo"; //~ ERROR: bare CR not allowed in raw string

    // the following string literal has a bare CR in it
    let _s = "foo\bar"; //~ ERROR: unknown character escape: \r
}
