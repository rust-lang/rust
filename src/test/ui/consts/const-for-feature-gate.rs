// gate-test-const_for

const _: () = {
    for _ in 0..5 {}
    //~^ error: `for` is not allowed in a `const`
};

fn main() {}
