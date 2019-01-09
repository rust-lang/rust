// Test that we use fully-qualified type names in error messages.

// ignore-test

type T1 = usize;
type T2 = isize;

fn bar(x: T1) -> T2 {
    return x;
    //~^ ERROR mismatched types: expected `T2`, found `T1`
}

fn main() {
}
