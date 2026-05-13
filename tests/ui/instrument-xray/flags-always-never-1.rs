// Checks that `-Z instrument-function=xray` does not allow `always` and `never` simultaneously.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray:always,never

fn main() {}

//~? ERROR incorrect value `xray:always,never` for unstable option `instrument-function`
