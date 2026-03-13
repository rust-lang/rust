// gate-test-const_for

const _: () = {
    for _ in 0..5 {}
    //~^ ERROR `std::ops::Range<{integer}>: const Iterator` is not satisfied
    //~| ERROR `std::ops::Range<{integer}>: const Iterator` is not satisfied
};

fn main() {}
