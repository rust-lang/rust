//@ check-fail
//@ edition:2018
//@ failure-status: 101
//@ aux-build:pub_trait.rs
//@ compile-flags: --test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

#![doc(test(attr(deny(non_local_definitions))))]
#![doc(test(attr(allow(dead_code))))]

/// This will produce a warning:
/// ```rust,no_run
/// # extern crate pub_trait;
/// # use pub_trait::Trait;
///
/// struct Local;
///
/// fn foo() {
///     impl Trait for &Local {}
/// }
/// ```
///
/// But this shouldn't produce a warning:
/// ```rust,no_run
/// # extern crate pub_trait;
/// # use pub_trait::Trait;
///
/// struct Local;
/// impl Trait for &Local {}
///
/// # fn main() {}
/// ```
pub fn doctest() {}
