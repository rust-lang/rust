//@require-annotations-for-level: WARN
#![allow(unused)]
#![warn(clippy::tests_outside_test_module)]

fn main() {
    // test code goes here
}

// Should lint
#[test]
fn my_test() {}
//~^ ERROR: this function marked with #[test] is outside a #[cfg(test)] module
//~| NOTE: move it to a testing module marked with #[cfg(test)]

#[cfg(test)]
mod tests {
    // Should not lint
    #[test]
    fn my_test() {}
}
