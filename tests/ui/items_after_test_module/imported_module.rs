//@ check-pass
//@compile-flags: --test
#![allow(unused)]
#![warn(clippy::items_after_test_module)]

// Nothing here should lint, as `tests` is an imported module (that has no body).

fn main() {}

fn should_not_lint() {}

#[path = "auxiliary/tests.rs"]
#[cfg(test)]
mod tests; // Should not lint

fn should_not_lint2() {}

const SHOULD_ALSO_NOT_LINT: usize = 1;
macro_rules! should_not_lint {
    () => {};
}
