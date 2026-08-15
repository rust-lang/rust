// #120427
// This test checks that when a single cfg has a value for user's specified name
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg(foo,values("my_value")) --check-cfg=cfg(bar,values("my_value"))

#[cfg(my_value)]
//~^ WARNING unexpected `cfg` condition name: `my_value`
fn x() {}

fn main() {}
