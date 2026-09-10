//! Regression test for https://github.com/rust-lang/rust/issues/162579

#[unsafe(rustc_allow_lifetime_dependent_specialization)]
trait A = B;
trait B {}
