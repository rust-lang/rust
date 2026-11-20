//@ check-pass
//@ no-auto-check-cfg
//@ revisions: cargo rustc
//@ [rustc]unset-rustc-env:CARGO_CRATE_NAME
//@ [cargo]rustc-env:CARGO_CRATE_NAME=foo
//@ compile-flags: --check-cfg=cfg(feature,values("foo")) --check-cfg=cfg(no_values)
//@ compile-flags: --check-cfg=cfg(quote,values("quote"))

#[cfg(featur)]
//~^ WARNING unexpected `cfg` condition name
fn feature() {}

#[cfg(featur = "foo")]
//~^ WARNING unexpected `cfg` condition name
fn feature() {}

#[cfg(featur = "fo")]
//~^ WARNING unexpected `cfg` condition name
fn feature() {}

#[cfg(feature = "foo")]
fn feature() {}

#[cfg(no_value)]
//~^ WARNING unexpected `cfg` condition name
fn no_values() {}

#[cfg(no_value = "foo")]
//~^ WARNING unexpected `cfg` condition name
fn no_values() {}

#[cfg(no_values = "bar")]
//~^ WARNING unexpected `cfg` condition value
fn no_values() {}

#[cfg(quote = "quote\"")]
//~^ WARNING unexpected `cfg` condition value
fn no_values() {}

fn main() {}
