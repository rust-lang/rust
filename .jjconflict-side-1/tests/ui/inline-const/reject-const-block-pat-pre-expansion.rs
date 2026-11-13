//! Regression test for <https://github.com/rust-lang/rust/issues/152499>: reject inline const
//! patterns pre-expansion when possible.

macro_rules! analyze { ($p:pat) => {}; }
analyze!(const { 0 });
//~^ ERROR:  const blocks cannot be used as patterns

#[cfg(false)]
fn scope() { let const { 0 }; }
//~^ ERROR: const blocks cannot be used as patterns

fn main() {}
