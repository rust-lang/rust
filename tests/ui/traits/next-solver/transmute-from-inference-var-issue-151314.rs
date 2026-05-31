//@ compile-flags: -Znext-solver=globally

// Regression test for #151314.

#![feature(transmutability)]

fn assert_transmutable<T>()
where
    (): std::mem::TransmuteFrom<T>,
{
}

fn main() {
    assert_transmutable()
    //~^ ERROR type annotations needed
}
