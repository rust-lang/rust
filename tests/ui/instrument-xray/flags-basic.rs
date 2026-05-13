// Verifies basic `-Z instrument-function=xray` options.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray
//@ compile-flags: -Z instrument-function=xray:skip-exit
//@ compile-flags: -Z instrument-function=xray:ignore-loops,instruction-threshold=300
//@ check-pass

fn main() {}
