//@ build-pass

#![crate_type = "cdylib"]

#[export_name = "foo.0123"]
pub extern "C" fn foo() {}

#[export_name = "EXPORTS"]
pub extern "C" fn bar() {}
