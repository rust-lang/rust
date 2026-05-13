#![crate_type = "lib"]

#[export_name = "\0foo"]
//~^ ERROR `export_name` may not contain null characters
fn has_null_byte() {}

#[export_name = "foo\0"]
//~^ ERROR `export_name` may not contain null characters
fn null_terminated() {}

#[export_name = "\0"]
//~^ ERROR `export_name` may not contain null characters
fn empty_null() {}

#[export_name = ""]
//~^ ERROR `export_name` may not be empty
fn empty() {}

#[export_name = "\
"]
//~^^ ERROR `export_name` may not be empty
fn empty_newline() {}
