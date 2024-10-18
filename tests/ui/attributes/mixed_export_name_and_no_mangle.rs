// issue: rust-lang/rust#47446
//@ run-rustfix
//@ check-pass

#![warn(unused_attributes)]
#[no_mangle]
//~^ WARN the `#[no_mangle]` attribute may not be used in combination with `#[export_name]`
#[export_name = "foo"]
pub fn bar() {}

fn main() {}
