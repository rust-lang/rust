// compile-flags:--test
// xfail-fast

extern mod std;

// Building as a test runner means that a synthetic main will be run,
// not ours
fn main() { fail; }
