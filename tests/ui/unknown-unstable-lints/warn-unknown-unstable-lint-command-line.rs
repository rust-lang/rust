//@ check-pass
//@ compile-flags: -Wunknown_lints -Atest_unstable_lint
//@ error-pattern: unknown lint: `test_unstable_lint`
//@ error-pattern: the `test_unstable_lint` lint is unstable

fn main() {}
