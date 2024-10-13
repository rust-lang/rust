#![deny(mixed_export_name_and_no_mangle)]
// issue: rust-lang/rust#47446

#[no_mangle]
#[export_name = "foo"]
pub fn bar() {}

fn main() {}
