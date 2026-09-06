//! Regression test for https://github.com/rust-lang/rust/issues/72323.
//! Block-tail constraints must not override coercions or unrelated inference.

trait Trait {}

#[derive(Default)]
struct Concrete {
    field: i32,
}

impl Trait for Concrete {}

fn coercion_target() {
    let _: Box<dyn Trait> = {
        let value = Default::default();
        //~^ ERROR type annotations needed
        let _ = value.field;
        value
    };
}

#[derive(Default)]
struct Returned;

fn unrelated_receiver() {
    let _: Returned = {
        let returned = Default::default();
        let receiver = Default::default();
        //~^ ERROR type annotations needed
        receiver.missing();
        returned
    };
}

fn non_tail_local() {
    let _: Returned = {
        let receiver = Default::default();
        //~^ ERROR type annotations needed
        receiver.missing();
        let returned = Default::default();
        returned
    };
}

fn cast_expectation() {
    let _ = {
        let value = Default::default();
        //~^ ERROR type annotations needed
        value.abs();
        value
    } as i32;
}

fn main() {}
