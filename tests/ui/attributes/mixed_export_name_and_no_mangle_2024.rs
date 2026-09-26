// issue: rust-lang/rust#47446
//@ run-rustfix
//@ edition:2024

#![warn(unused_attributes)]
#[unsafe(no_mangle)]
//~^ ERROR `#[unsafe(no_mangle)]` attribute may not be used in combination with `#[unsafe(export_name)]` [harmful_unused_attributes]
#[unsafe(export_name = "foo")]
pub fn bar() {}

#[unsafe(no_mangle)]
//~^ ERROR `#[unsafe(no_mangle)]` attribute may not be used in combination with `#[unsafe(export_name)]` [harmful_unused_attributes]
#[unsafe(export_name = "baz")]
pub fn bak() {}

fn main() {}
