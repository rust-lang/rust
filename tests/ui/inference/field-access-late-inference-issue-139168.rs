//! Regression test for https://github.com/rust-lang/rust/issues/139168.
//! The binding is returned directly, but field access requires its type earlier.

use std::collections::BTreeMap;

#[derive(Default)]
struct Something {
    value: i32,
}

fn later_known() -> Something {
    let mut value = Default::default();
    //~^ ERROR type annotations needed
    value.value = 100;
    value
}

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

fn generic_adt() -> BTreeMap<i32, Vec<i32>> {
    let mut result = BTreeMap::new();
    //~^ ERROR type annotations needed
    result.entry(1).or_default().push(1);
    result
}

fn main() {
    let _ = later_known();
}
