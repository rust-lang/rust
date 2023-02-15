// Checks that `-Z instrument-xray` does not allow `always` and `never` simultaneously.
//
// needs-xray
// compile-flags: -Z instrument-xray=always,never
// error-pattern: incorrect value `always,never` for unstable option `instrument-xray`

fn main() {}
