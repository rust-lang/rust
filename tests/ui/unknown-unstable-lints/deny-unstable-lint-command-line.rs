// check-fail
//@compile-flags: -Dunknown_lints -Atest_unstable_lint
//@error-in-other-file: unknown lint: `test_unstable_lint`
//@error-in-other-file: the `test_unstable_lint` lint is unstable

fn main() {}
