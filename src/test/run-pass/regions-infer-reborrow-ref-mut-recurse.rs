// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test an edge case in region inference: the lifetime of the borrow
// of `*x` must be extended to at least 'a.

fn foo<'a,'b>(x: &'a &'b mut int) -> &'a int {
    let y = &*x; // should be inferred to have type &'a &'b mut int...

    // ...because if we inferred, say, &'x &'b mut int where 'x <= 'a,
    // this reborrow would be illegal:
    &**y
}

pub fn main() {
    /* Just want to know that it compiles. */
}
