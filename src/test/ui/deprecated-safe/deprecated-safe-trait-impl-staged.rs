// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(negative_impls)]
#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(deprecated_safe_in_future, unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{DeprSafe, DeprSafe2015, DeprSafe2015Future, DeprSafe2018, DeprSafeFuture};

struct A;
impl DeprSafe for A {} //~ WARN use of trait `deprecated_safe::DeprSafe` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct B;
unsafe impl DeprSafe for B {}

struct C;
impl !DeprSafe for C {}

struct D;
unsafe impl !DeprSafe for D {} //~ ERROR negative impls cannot be unsafe

struct E;
impl DeprSafeFuture for E {} //~ WARN use of trait `deprecated_safe::DeprSafeFuture` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct F;
unsafe impl DeprSafeFuture for F {}

struct G;
impl !DeprSafeFuture for G {}

struct H;
unsafe impl !DeprSafeFuture for H {} //~ ERROR negative impls cannot be unsafe

struct I;
impl DeprSafe2015 for I {} //~ ERROR the trait `DeprSafe2015` requires an `unsafe impl` declaration

struct J;
unsafe impl DeprSafe2015 for J {}

struct K;
impl !DeprSafe2015 for K {}

struct L;
unsafe impl !DeprSafe2015 for L {} //~ ERROR negative impls cannot be unsafe

struct M;
impl DeprSafe2015Future for M {} //~ WARN use of trait `deprecated_safe::DeprSafe2015Future` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct N;
unsafe impl DeprSafe2015Future for N {}

struct O;
impl !DeprSafe2015Future for O {}

struct P;
unsafe impl !DeprSafe2015Future for P {} //~ ERROR negative impls cannot be unsafe

struct Q;
impl DeprSafe2018 for Q {} //~ WARN use of trait `deprecated_safe::DeprSafe2018` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct R;
unsafe impl DeprSafe2018 for R {}

struct S;
impl !DeprSafe2018 for S {}

struct T;
unsafe impl !DeprSafe2018 for T {} //~ ERROR negative impls cannot be unsafe

fn main() {}
