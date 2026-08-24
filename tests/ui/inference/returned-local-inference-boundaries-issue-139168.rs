//! Regression test for https://github.com/rust-lang/rust/issues/139168.
//! Return-type constraints must preserve coercions and unrelated inference failures.

trait Trait {}

#[derive(Default)]
struct Concrete {
    field: i32,
}

impl Trait for Concrete {}

fn coercion_target() -> Box<dyn Trait> {
    let value = Default::default();
    //~^ ERROR type annotations needed
    let _ = value.field;
    value
}

#[derive(Default)]
struct Returned;

fn unrelated_receiver() -> Returned {
    let returned = Default::default();
    let receiver = Default::default();
    //~^ ERROR type annotations needed
    receiver.missing();
    returned
}

fn returned_local_not_gathered_yet() -> Returned {
    let receiver = Default::default();
    //~^ ERROR type annotations needed
    receiver.missing();
    let returned = Default::default();
    returned
}

fn main() {}
