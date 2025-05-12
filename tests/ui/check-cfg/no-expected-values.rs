// Check that we detect unexpected value when none are allowed
//
//@ check-pass
//@ no-auto-check-cfg
//@ revisions: simple mixed empty
//@ compile-flags: --check-cfg=cfg(values,simple,mixed,empty)
//@ [simple]compile-flags: --check-cfg=cfg(test) --check-cfg=cfg(feature)
//@ [mixed]compile-flags: --check-cfg=cfg(test,feature)
//@ [empty]compile-flags: --check-cfg=cfg(test,feature,values(none()))

#[cfg(feature = "foo")]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

#[cfg(test = "foo")]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

fn main() {}
