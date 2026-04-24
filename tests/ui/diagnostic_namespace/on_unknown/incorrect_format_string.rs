#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "foo {}")]
//~^ WARN: positional arguments are not permitted in diagnostic attributes
use std::does_not_exist;
//~^ ERROR: foo {}

#[diagnostic::on_unknown(message = "foo {A}")]
//~^ WARN: this format argument is not allowed in `#[diagnostic::on_unknown]`
use std::does_not_exist2;
//~^ ERROR: foo {A}

#[diagnostic::on_unknown(label = "foo {}")]
//~^ WARN: positional arguments are not permitted in diagnostic attributes
use std::does_not_exist3;
//~^ ERROR: unresolved import `std::does_not_exist3`

#[diagnostic::on_unknown(label = "foo {A}")]
//~^ WARN: this format argument is not allowed in `#[diagnostic::on_unknown]`
use std::does_not_exist4;
//~^ ERROR: unresolved import `std::does_not_exist4`

#[diagnostic::on_unknown(note = "foo {}")]
//~^ WARN: positional arguments are not permitted in diagnostic attributes
use std::does_not_exist5;
//~^ ERROR: unresolved import `std::does_not_exist5`

#[diagnostic::on_unknown(note = "foo {Self} {ItemContext} {Trait} {A}")]
//~^ WARN: this format argument is not allowed in `#[diagnostic::on_unknown]`
//~| WARN: this format argument is not allowed in `#[diagnostic::on_unknown]`
//~| WARN: this format argument is not allowed in `#[diagnostic::on_unknown]`
//~| WARN: this format argument is not allowed in `#[diagnostic::on_unknown]`
use std::does_not_exist6;
//~^ ERROR: unresolved import `std::does_not_exist6`

fn main() {}
