// Checks that `-Z instrument-function=xray` does not allow duplicate options.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray:always,always

fn main() {}

//~? ERROR incorrect value `xray:always,always` for unstable option `instrument-function`
