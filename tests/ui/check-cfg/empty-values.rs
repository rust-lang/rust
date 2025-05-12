// Check that we detect unexpected value when none are allowed
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg(foo,values())

#[cfg(foo = "foo")]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

#[cfg(foo)]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

fn main() {}
