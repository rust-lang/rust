// Checks that `-Z instrument-xray` allows `always` and `never` sequentially.
// (The last specified setting wins, like `-Z instrument-xray=no` as well.)
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray=always
//@ compile-flags: -Z instrument-xray=never
//@ check-pass

fn main() {}
