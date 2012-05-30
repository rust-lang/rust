// aux-build:cci_capture_clause.rs
// xfail-fast

// This test makes sure we can do cross-crate inlining on functions
// that use capture clauses.

use cci_capture_clause;

import comm::recv;
import comm::methods;

fn main() {
    cci_capture_clause::foo(()).recv()
}
