//@ compile-flags: -Znext-solver=globally

// Regression test for https://github.com/rust-lang/rust/issues/151957

trait PointeeSized {
    type Undefined;
}

struct T;

impl PointeeSized for T
//~^ ERROR not all trait items implemented, missing: `Undefined`
//~| ERROR the trait bound `T: PointeeSized` is not satisfied
where
    <T as PointeeSized>::Undefined: PointeeSized,
//~^ ERROR the trait bound `T: PointeeSized` is not satisfied
//~| ERROR the trait bound `T: PointeeSized` is not satisfied
{}

fn main() {}
