// Regression test for issue #55825
// Tests that we don't emit a spurious warning in NLL mode

#![feature(nll)]

const fn no_dyn_trait_ret() -> &'static dyn std::fmt::Debug { &() } //~ ERROR const

fn main() { }
