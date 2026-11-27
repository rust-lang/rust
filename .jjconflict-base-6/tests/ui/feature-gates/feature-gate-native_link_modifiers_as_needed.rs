//@ revisions: in_attr in_flag
//@[in_flag] compile-flags: -l dylib:+as-needed=foo

#[cfg(in_attr)]
#[link(name = "foo", kind = "dylib", modifiers = "+as-needed")]
//[in_attr]~^ ERROR: linking modifier `as-needed` is unstable
extern "C" {}

fn main() {}

//[in_flag]~? ERROR linking modifier `as-needed` is unstable
