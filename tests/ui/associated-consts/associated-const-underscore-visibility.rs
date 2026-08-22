#![feature(associated_const_underscore)]
#![deny(private_interfaces)]
fn main() {}

pub struct Public;
struct Private;

impl Public {
    pub const _: Private = Private;
    //~^ ERROR: type `Private` is more private than the item `Public::_` [private_interfaces]
}

impl Public {
    const _: Private = Private;
}

impl Private {
    pub const _: Private = Private;
    pub const _: () = {};
}
