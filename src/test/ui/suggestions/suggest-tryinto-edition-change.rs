// Make sure that calling `.try_into()` in pre-2021 mentions Edition 2021 change
// edition:2018

fn test() {
    let i: i16 = 0_i32.try_into().unwrap();
    //~^ ERROR no method named `try_into` found for type `i32` in the current scope
    //~| NOTE method not found in `i32`
    //~| NOTE 'std::convert::TryInto' is included in the prelude starting in Edition 2021
}

fn main() {}
