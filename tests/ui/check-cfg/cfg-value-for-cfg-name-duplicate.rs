// #120427
// This test checks we won't suggest more than 3 span suggestions for cfg names
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg(foo,values("value")) --check-cfg=cfg(bar,values("value")) --check-cfg=cfg(bee,values("value")) --check-cfg=cfg(cow,values("value"))

#[cfg(value)]
//~^ WARNING unexpected `cfg` condition name: `value`
fn x() {}

fn main() {}
