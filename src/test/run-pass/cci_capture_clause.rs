// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cci_capture_clause.rs

// This test makes sure we can do cross-crate inlining on functions
// that use capture clauses.

extern crate cci_capture_clause;

pub fn main() {
    cci_capture_clause::foo(()).recv().unwrap();
}
