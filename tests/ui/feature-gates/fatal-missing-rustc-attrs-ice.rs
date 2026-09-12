//! Regression test for https://github.com/rust-lang/rust/issues/162579

#[unsafe(rustc_allow_lifetime_dependent_specialization)] //~ ERROR: use of an internal attribute
trait A = B; //~ ERROR: trait aliases are experimental
trait B {}
