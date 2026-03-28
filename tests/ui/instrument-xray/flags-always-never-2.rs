// Checks that `-Z instrument-xray-opts` allows `always` and `never` sequentially.
// (The last specified setting wins)
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray-opts=always
//@ compile-flags: -Z instrument-xray-opts=never
//@ check-pass

fn main() {}
