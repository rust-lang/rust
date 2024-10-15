// issue: rust-lang/rust#47446
//@ run-rustfix
//@ check-pass

#[no_mangle]
#[export_name = "foo"]
//~^ WARN the attribute `export_name` may not be used in combination with `no_mangle`
pub fn bar() {}

fn main() {}
