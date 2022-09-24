// Checks that `-Z instrument-xray` does not allow duplicates.
//
// compile-flags: -Z instrument-xray=ignore-loops,ignore-loops
// error-pattern: incorrect value `ignore-loops,ignore-loops` for unstable option `instrument-xray`

fn main() {}
