// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

//@ compile-flags:--test
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

/// ```compile_fail
/// println!("Hello");
/// ```
pub struct Foo;
