#![feature(diagnostic_on_unknown_item)]

#[diagnostic::on_unknown_item(message = "foo {}")]
//~^ WARN: format arguments are not allowed here
use std::does_not_exist;
//~^ ERROR: foo {}

#[diagnostic::on_unknown_item(message = "foo {A}")]
//~^ WARN: format arguments are not allowed here
use std::does_not_exist2;
//~^ ERROR: foo {}

#[diagnostic::on_unknown_item(label = "foo {}")]
//~^ WARN: format arguments are not allowed here
use std::does_not_exist3;
//~^ ERROR: unresolved import `std::does_not_exist3`

#[diagnostic::on_unknown_item(label = "foo {A}")]
//~^ WARN: format arguments are not allowed here
use std::does_not_exist4;
//~^ ERROR: unresolved import `std::does_not_exist4`

#[diagnostic::on_unknown_item(note = "foo {}")]
//~^ WARN: format arguments are not allowed here
use std::does_not_exist5;
//~^ ERROR: unresolved import `std::does_not_exist5`

#[diagnostic::on_unknown_item(note = "foo {A}")]
//~^ WARN: format arguments are not allowed here
use std::does_not_exist6;
//~^ ERROR: unresolved import `std::does_not_exist6`

fn main() {}
