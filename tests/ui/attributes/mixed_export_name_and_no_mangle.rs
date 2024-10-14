// issue: rust-lang/rust#47446

#![deny(mixed_export_name_and_no_mangle)]

#[no_mangle]
#[export_name = "foo"]
//~^ ERROR the attribute `export_name` may not be used in combination with `no_mangle`
pub fn bar() {}

fn main() {}
