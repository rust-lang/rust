// gate-test-const_for

const _: () = {
    for _ in 0..5 {}
    //~^ error: `for` is not allowed in a `const`
    //~| ERROR: cannot convert
    //~| ERROR: cannot call
    //~| ERROR: mutable references
};

fn main() {}
