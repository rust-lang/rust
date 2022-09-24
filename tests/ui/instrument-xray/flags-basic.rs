// Verifies basic `-Z instrument-xray` flags.
//
// compile-flags: -Z instrument-xray
// compile-flags: -Z instrument-xray=skip-exit
// compile-flags: -Z instrument-xray=ignore-loops,instruction-threshold=300
// check-pass

fn main() {}
