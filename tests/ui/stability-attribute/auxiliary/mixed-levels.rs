//! definitions for ../mixed-levels.rs

#![stable(feature = "stable_feature", since = "1.0.0")]
#![feature(staged_api)]
#![crate_type = "lib"]

#[stable(feature = "stable_a", since = "1.0.0")]
#[stable(feature = "stable_b", since = "1.8.2")]
#[macro_export]
macro_rules! stable_mac {
    () => ()
}

#[unstable(feature = "unstable_a", issue = "none")]
#[stable(feature = "stable_a", since = "1.0.0")]
#[macro_export]
macro_rules! unstable_mac {
    () => ()
}

#[stable(feature = "stable_feature", since = "1.0.0")]
#[rustc_const_stable(feature = "stable_c", since = "1.8.2")]
#[rustc_const_stable(feature = "stable_d", since = "1.0.0")]
pub const fn const_stable_fn() {}

#[stable(feature = "stable_feature", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable_c", issue = "none")]
#[rustc_const_stable(feature = "stable_c", since = "1.8.2")]
pub const fn const_unstable_fn() {}
