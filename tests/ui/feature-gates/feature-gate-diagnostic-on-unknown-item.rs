#![deny(warnings)]

#[diagnostic::on_unknown_item(message = "Tada")]
//~^ ERROR: unknown diagnostic attribute
use std::vec::NotExisting;
//~^ ERROR: unresolved import `std::vec::NotExisting`

fn main() {}
