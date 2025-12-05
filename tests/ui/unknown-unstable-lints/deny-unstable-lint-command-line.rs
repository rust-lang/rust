//~ ERROR unknown lint: `test_unstable_lint`
//~^ NOTE the `test_unstable_lint` lint is unstable
//@ check-fail
//@ compile-flags: -Dunknown_lints -Atest_unstable_lint
//@ dont-require-annotations: NOTE

fn main() {}
