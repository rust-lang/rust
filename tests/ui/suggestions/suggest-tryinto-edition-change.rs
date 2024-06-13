// Make sure that trying to access `TryInto`, `TryFrom`, `FromIterator` in pre-2021 mentions
// Edition 2021 change.
//@ edition:2018

fn test() {
    let _i: i16 = 0_i32.try_into().unwrap();
    //~^ ERROR no method named `try_into` found for type `i32` in the current scope
    //~| NOTE 'std::convert::TryInto' is included in the prelude starting in Edition 2021

    let _i: i16 = TryFrom::try_from(0_i32).unwrap();
    //~^ ERROR failed to resolve: use of undeclared type
    //~| NOTE use of undeclared type
    //~| NOTE 'std::convert::TryFrom' is included in the prelude starting in Edition 2021

    let _i: i16 = TryInto::try_into(0_i32).unwrap();
    //~^ ERROR failed to resolve: use of undeclared type
    //~| NOTE use of undeclared type
    //~| NOTE 'std::convert::TryInto' is included in the prelude starting in Edition 2021

    let _v: Vec<_> = FromIterator::from_iter(&[1]);
    //~^ ERROR failed to resolve: use of undeclared type
    //~| NOTE use of undeclared type
    //~| NOTE 'std::iter::FromIterator' is included in the prelude starting in Edition 2021
}

fn main() {
    test();
}
