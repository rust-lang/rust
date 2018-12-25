// run-pass
// aux-build:dynamic_runner.rs
// compile-flags:--test
#![feature(custom_test_frameworks)]
#![test_runner(dynamic_runner::runner)]

extern crate dynamic_runner;

pub struct AllFoo(&'static str);
struct IsFoo(String);

impl dynamic_runner::Testable for AllFoo {
    fn name(&self) -> String {
        String::from(self.0)
    }

    fn subtests(&self) -> Vec<Box<dyn dynamic_runner::Testable>> {
        self.0.split(" ").map(|word|
            Box::new(IsFoo(word.into())) as Box<dyn dynamic_runner::Testable>
        ).collect()
    }
}

impl dynamic_runner::Testable for IsFoo {
    fn name(&self) -> String {
        self.0.clone()
    }

    fn run(&self) -> bool {
        self.0 == "foo"
    }
}

#[test_case]
const TEST_2: AllFoo = AllFoo("foo foo");
