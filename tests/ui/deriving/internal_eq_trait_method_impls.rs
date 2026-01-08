#![deny(deprecated, internal_eq_trait_method_impls)]
pub struct Bad;

impl PartialEq for Bad {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Bad {
    fn assert_receiver_is_total_eq(&self) {}
    //~^ ERROR: `Eq::assert_receiver_is_total_eq` should never be implemented by hand [internal_eq_trait_method_impls]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

#[derive(PartialEq, Eq)]
pub struct Good;

#[derive(PartialEq)]
pub struct Good2;

impl Eq for Good2 {}

pub struct Foo;

pub trait SameName {
    fn assert_receiver_is_total_eq(&self) {}
}

impl SameName for Foo {
    fn assert_receiver_is_total_eq(&self) {}
}

pub fn main() {
    Foo.assert_receiver_is_total_eq();
    Good2.assert_receiver_is_total_eq();
    //~^ ERROR: use of deprecated method `std::cmp::Eq::assert_receiver_is_total_eq`: implementation detail of `#[derive(Eq)]` [deprecated]
    Good.assert_receiver_is_total_eq();
    //~^ ERROR: use of deprecated method `std::cmp::Eq::assert_receiver_is_total_eq`: implementation detail of `#[derive(Eq)]` [deprecated]
    Bad.assert_receiver_is_total_eq();
    //~^ ERROR: use of deprecated method `std::cmp::Eq::assert_receiver_is_total_eq`: implementation detail of `#[derive(Eq)]` [deprecated]
}
