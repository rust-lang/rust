#![allow(unused)]
#![warn(clippy::tests_outside_test_module)]

fn main() {
    // test code goes here
}

// Should lint
#[test]
fn my_test() {}

#[cfg(test)]
mod tests {
    // Should not lint
    #[test]
    fn my_test() {}
}
