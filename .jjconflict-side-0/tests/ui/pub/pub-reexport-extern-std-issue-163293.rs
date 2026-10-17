pub use std as x;
//~^ ERROR extern crate `std` is private and cannot be re-exported
//~^^ WARN this was previously accepted by the compiler but is being phased out

//@ edition: 2015

fn main() {}
