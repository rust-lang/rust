//@ compile-flags: -Znext-solver=globally

fn main() {
    const {
        for _ in 1..5 {}
        //~^ ERROR cannot use `for` loop on `std::ops::Range<i32>` in constants
        //~| ERROR `IntoIterator` is not yet stable as a const trait
        //~| ERROR cannot use `for` loop on `std::ops::Range<i32>` in constants
        //~| ERROR `Iterator` is not yet stable as a const trait
    }
}
