// Check that we detect unexpected value when none are allowed
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg(foo,values()) -Zunstable-options

#[cfg(foo = "foo")]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

#[cfg(foo)]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

fn main() {}
