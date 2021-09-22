// Regression test for the ICE described in #79546.

// compile-flags: --cap-lints=allow -Zfuel=issue79546=0
// check-pass
#![crate_name="issue79546"]

struct S;
fn main() {}
