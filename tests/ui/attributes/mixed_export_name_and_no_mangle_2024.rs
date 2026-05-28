// issue: rust-lang/rust#47446
//@ run-rustfix
//@ check-pass
//@ edition:2024

#![warn(unused_attributes)]
#[unsafe(no_mangle)]
//~^ WARN `#[unsafe(no_mangle)]` attribute may not be used in combination with `#[unsafe(export_name)]` [unused_attributes]
#[unsafe(export_name = "foo")]
pub fn bar() {}

#[unsafe(no_mangle)]
//~^ WARN `#[unsafe(no_mangle)]` attribute may not be used in combination with `#[unsafe(export_name)]` [unused_attributes]
#[unsafe(export_name = "baz")]
pub fn bak() {}

fn main() {}
