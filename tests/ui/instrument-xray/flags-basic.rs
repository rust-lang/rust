// Verifies basic `-Z instrument-xray` flags.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray
//@ compile-flags: -Z instrument-function=xray -Z instrument-xray-opts=skip-exit
//@ compile-flags: -Z instrument-function=xray -Z instrument-xray-opts=ignore-loops,instruction-threshold=300
//@ check-pass

fn main() {}
