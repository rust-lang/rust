// This test makes sure that we detect changed feature gates.

//@ revisions:rpass1 cfail2
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![cfg_attr(rpass1, feature(abi_unadjusted))]

fn main() {
}

extern "unadjusted" fn foo() {}
//[cfail2]~^ ERROR: "unadjusted" ABI is an implementation detail and perma-unstable
