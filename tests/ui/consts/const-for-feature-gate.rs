// gate-test-const_for

const _: () = {
    for _ in 0..5 {}
    //~^ ERROR cannot use `for`
    //~| ERROR cannot use `for`
};

fn main() {}
