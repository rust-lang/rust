//@ compile-flags: -Znext-solver=globally
//@ edition: 2021

fn main() {
    const {
        for _ in 1..5 {}
        //~^ ERROR cannot use `for` loop
        //~| ERROR `IntoIterator` is not yet stable
        //~| ERROR cannot use `for` loop
        //~| ERROR `Iterator` is not yet stable
    }
}
