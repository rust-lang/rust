// Check that we detect unexpected value when none are allowed
//
// check-pass
// compile-flags: --check-cfg=values(test) --check-cfg=values(feature) -Z unstable-options

#[cfg(feature = "foo")]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

#[cfg(test = "foo")]
//~^ WARNING unexpected `cfg` condition value
fn do_foo() {}

fn main() {}
