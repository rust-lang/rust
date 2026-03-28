// Checks that `-Z instrument-xray-opts` does not allow `always` and `never` simultaneously.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray -Z instrument-xray-opts=always,never

fn main() {}

//~? ERROR incorrect value `always,never` for unstable option `instrument-xray-opts`
