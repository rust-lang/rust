// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{DeprSafe2015, DeprSafe2015Future, DeprSafe2018};

struct DeprSafeImpl;
impl DeprSafe2015 for DeprSafeImpl {} //~ ERROR the trait `DeprSafe2015` requires an `unsafe impl` declaration
impl DeprSafe2015Future for DeprSafeImpl {}
impl DeprSafe2018 for DeprSafeImpl {} //~ WARN use of trait `deprecated_safe::DeprSafe2018` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct DeprSafeUnsafeImpl;
unsafe impl DeprSafe2015 for DeprSafeUnsafeImpl {}
unsafe impl DeprSafe2015Future for DeprSafeUnsafeImpl {} //~ ERROR implementing the trait `DeprSafe2015Future` is not unsafe
unsafe impl DeprSafe2018 for DeprSafeUnsafeImpl {}

fn main() {
    // NOTE: this test is separate from deprecated-safe-unsafe-edition as the other compiler
    // errors will stop compilation before these calls are checked
}
