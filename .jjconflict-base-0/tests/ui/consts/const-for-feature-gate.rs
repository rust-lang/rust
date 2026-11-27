// gate-test-const_for

const _: () = {
    for _ in 0..5 {}
    //~^ ERROR cannot use `for`
    //~| ERROR `IntoIterator` is not yet stable
    //~| ERROR cannot use `for`
    //~| ERROR `Iterator` is not yet stable
};

fn main() {}
