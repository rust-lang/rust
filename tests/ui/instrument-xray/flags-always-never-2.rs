// Checks that `-Z instrument-function=xray` allows `always` and `never` sequentially.
// (The last specified setting wins)
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray:always
//@ compile-flags: -Z instrument-function=xray:never
//@ check-pass

fn main() {}
