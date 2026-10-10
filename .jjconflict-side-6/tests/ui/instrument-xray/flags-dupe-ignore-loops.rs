// Checks that `-Z instrument-xray` does not allow duplicates.
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray=ignore-loops,ignore-loops

fn main() {}

//~? ERROR incorrect value `ignore-loops,ignore-loops` for unstable option `instrument-xray`
