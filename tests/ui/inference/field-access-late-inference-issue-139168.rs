//! Regression test for https://github.com/rust-lang/rust/issues/139168.
//! The binding is returned directly, but earlier uses require its type.

//@ check-pass

use std::collections::BTreeMap;

#[derive(Default)]
struct Something {
    value: i32,
}

fn later_known() -> Something {
    let mut value = Default::default();
    value.value = 100;
    value
}

fn generic_adt() -> BTreeMap<i32, Vec<i32>> {
    let mut result = BTreeMap::new();
    result.entry(1).or_default().push(1);
    result
}

fn indexed() -> Vec<i32> {
    let mut values = Default::default();
    values[0] = 1;
    values
}

fn boxed() -> Box<Vec<i32>> {
    let mut value = Default::default();
    value.push(1);
    value
}

fn main() {
    let _ = later_known();
}
