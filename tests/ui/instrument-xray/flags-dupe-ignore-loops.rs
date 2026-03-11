// Checks that `-Z instrument-xray-opts` does not allow duplicates.
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray-opts=ignore-loops,ignore-loops

fn main() {}

//~? ERROR incorrect value `ignore-loops,ignore-loops` for unstable option `instrument-xray-opts`
