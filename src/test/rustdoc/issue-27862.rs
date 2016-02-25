// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check-search-index

/// Test  | Table
/// ------|-------------
/// t = b | id = \|x\| x
pub struct Foo; // @has issue_27862/struct.Foo.html //td 'id = |x| x'

/* !search-index
{
    "issue_27862": {
        "issue_27862::Foo": [
            "Struct",
            "Test  | Table\n------|-------------\nt = b | id = \\|x\\| x"
        ]
    }
}
*/
