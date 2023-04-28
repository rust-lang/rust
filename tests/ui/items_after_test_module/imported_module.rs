//@compile-flags: --test
#![allow(unused)]
#![warn(clippy::items_after_test_module)]

fn main() {}

fn should_not_lint() {}

#[path = "auxiliary/tests.rs"]
#[cfg(test)]
mod tests; // Should not lint

fn should_lint() {}

const SHOULD_ALSO_LINT: usize = 1;
macro_rules! should_not_lint {
    () => {};
}
