// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that failing to syntax highlight a rust code-block doesn't cause
// rustdoc to fail, while still rendering the code-block (without highlighting).


// @has issue_12834/fn.foo.html
// @has - //pre 'a + b '

/// ```
/// a + b ∈ Self ∀ a, b ∈ Self
/// ```
pub fn foo() {}
