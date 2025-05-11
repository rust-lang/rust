// Checks that `-Z instrument-xray` does not allow `always` and `never` simultaneously.
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray=always,never

fn main() {}

//~? ERROR incorrect value `always,never` for unstable option `instrument-xray`
