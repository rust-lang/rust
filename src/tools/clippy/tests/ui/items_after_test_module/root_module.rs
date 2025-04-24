#![warn(clippy::items_after_test_module)]

fn main() {}

fn should_not_lint() {}

#[allow(dead_code)]
#[allow(unused)] // Some attributes to check that span replacement is good enough
#[allow(clippy::allow_attributes)]
#[cfg(test)]
mod tests {
    //~^ items_after_test_module
    #[test]
    fn hi() {}
}

fn should_lint() {}

const SHOULD_ALSO_LINT: usize = 1;
macro_rules! should_lint {
    () => {};
}
