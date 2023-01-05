// run-pass
// aux-build:example_runner.rs
// compile-flags:--test

#![feature(custom_test_frameworks)]
#![test_runner(example_runner::runner)]
extern crate example_runner;

pub struct IsFoo(&'static str);

impl example_runner::Testable for IsFoo {
    fn name(&self) -> String {
        self.0.to_string()
    }

    fn run(&self) -> Option<String> {
        if self.0 != "foo" {
            return Some(format!("{} != foo", self.0));
        }
        None
    }
}

#[test_case]
const TEST_1: IsFoo = IsFoo("hello");

#[test_case]
const TEST_2: IsFoo = IsFoo("foo");
