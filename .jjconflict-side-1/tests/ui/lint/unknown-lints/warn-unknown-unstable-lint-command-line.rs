//~ WARN unknown lint: `test_unstable_lint`
//~^ NOTE the `test_unstable_lint` lint is unstable
//@ check-pass
//@ compile-flags: -Wunknown_lints -Atest_unstable_lint
//@ dont-require-annotations: NOTE

fn main() {}
