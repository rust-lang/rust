// Regression test for issue #55825
// Tests that we don't emit a spurious warning in NLL mode

//@ check-pass

const fn no_dyn_trait_ret() -> &'static dyn std::fmt::Debug { &() }

fn main() { }
