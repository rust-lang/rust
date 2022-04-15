// aux-build:deprecated-safe.rs
// revisions: mir thir
// NOTE(skippy) these tests output many duplicates, so deduplicate or they become brittle to changes
// [mir]compile-flags: -Zdeduplicate-diagnostics=yes
// [thir]compile-flags: -Z thir-unsafeck -Zdeduplicate-diagnostics=yes

#![feature(negative_impls)]
#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{DeprSafe, DeprSafe2015, DeprSafe2018};

struct A;
impl DeprSafe for A {} //~ WARN use of trait `deprecated_safe::DeprSafe` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct B;
unsafe impl DeprSafe for B {}

struct C;
impl !DeprSafe for C {}

struct D;
unsafe impl !DeprSafe for D {} //~ ERROR negative impls cannot be unsafe

struct I;
impl DeprSafe2015 for I {} //~ ERROR the trait `DeprSafe2015` requires an `unsafe impl` declaration

struct J;
unsafe impl DeprSafe2015 for J {}

struct K;
impl !DeprSafe2015 for K {}

struct L;
unsafe impl !DeprSafe2015 for L {} //~ ERROR negative impls cannot be unsafe

struct M;
impl DeprSafe2018 for M {} //~ WARN use of trait `deprecated_safe::DeprSafe2018` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct R;
unsafe impl DeprSafe2018 for R {}

struct S;
impl !DeprSafe2018 for S {}

struct T;
unsafe impl !DeprSafe2018 for T {} //~ ERROR negative impls cannot be unsafe

fn main() {}
