// Checks that `-Z instrument-function=xray` does not allow duplicate configuration options.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray:ignore-loops,ignore-loops

fn main() {}

//~? ERROR incorrect value `xray:ignore-loops,ignore-loops` for unstable option `instrument-function`
