// compile-flags:--test
// xfail-fast

use std;

// Building as a test runner means that a synthetic main will be run,
// not ours
fn main() {
    fail;
}