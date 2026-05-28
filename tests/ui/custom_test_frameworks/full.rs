//@ run-pass
//@ aux-build:example_runner.rs
//@ compile-flags:--test

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
static TEST_2: IsFoo = IsFoo("foo");

// FIXME: `test_case` is currently ignored on anything other than
// fn/const/static. This should be an error. Compare this with `#[test]` and
// #[bench] whose expanders emit "error: expected a non-associated function,
// found […]" if applied to invalid items.
#[test_case]
struct _S;

// FIXME: as above.
#[test_case]
impl _S {
    fn _f() {}
}
