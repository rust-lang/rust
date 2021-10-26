// Make sure that trying to access `TryInto`, `TryFrom`, `FromIterator` in pre-2021 mentions
// Edition 2021 change
// edition:2018

// We mark this no_std to avoid emitting suggestions for both `std` and `core` traits. These were
// inconsistently ordered between CI and at least one local build, causing test failures.
#![no_std]
#![crate_type = "lib"]

pub fn test() {
    let _i: Result<i16, _> = 0_i32.try_into();
    //~^ ERROR no method named `try_into` found for type `i32` in the current scope
    //~| NOTE method not found in `i32`
    //~| NOTE 'core::convert::TryInto' is included in the prelude starting in Edition 2021

    let _i: Result<i16, _> = TryFrom::try_from(0_i32);
    //~^ ERROR failed to resolve: use of undeclared type
    //~| NOTE not found in this scope
    //~| NOTE 'core::convert::TryFrom' is included in the prelude starting in Edition 2021

    let _i: Result<i16, _> = TryInto::try_into(0_i32);
    //~^ ERROR failed to resolve: use of undeclared type
    //~| NOTE not found in this scope
    //~| NOTE 'core::convert::TryInto' is included in the prelude starting in Edition 2021

    let _i: () = FromIterator::from_iter(core::iter::empty());
    //~^ ERROR failed to resolve: use of undeclared type
    //~| NOTE 'core::iter::FromIterator' is included in the prelude starting in Edition 2021
}
