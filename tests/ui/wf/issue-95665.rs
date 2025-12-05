// Regression test for the ICE described in #95665.
// Ensure that the expected error is output (and thus that there is no ICE)

pub trait Trait: {}

pub struct Struct<T: Trait> {
    member: T,
}

// uncomment and bug goes away
// impl Trait for u8 {}

extern "C" {
    static VAR: Struct<u8>;
    //~^ ERROR the trait bound `u8: Trait` is not satisfied [E0277]
}

fn main() {}
