// Checks that `-Z instrument-xray-opts` does not allow duplicates.
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray-opts=always,always

fn main() {}

//~? ERROR incorrect value `always,always` for unstable option `instrument-xray-opts`
