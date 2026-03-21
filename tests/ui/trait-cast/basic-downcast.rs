//@ run-pass
//! Basic trait-cast run-pass test: two concrete types, three sub-traits of a
//! common root.  Greet and Count are implemented for both types; Describe is
//! implemented only for TypeA.  A `#[inline(never)]` function receives
//! `&dyn Base` and attempts all three downcasts, exercising both the success
//! and failure paths.

#![feature(trait_cast)]

#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait Base: TraitMetadataTable<dyn Base> + core::fmt::Debug {
    fn name(&self) -> &'static str;
}

trait Greet: Base {
    fn greeting(&self) -> &'static str;
}

trait Count: Base {
    fn count(&self) -> u32;
}

trait Describe: Base {
    fn description(&self) -> &'static str;
}

// ---- concrete types ----
#[derive(Debug)]
struct TypeA;
#[derive(Debug)]
struct TypeB;

// Base — both types
impl Base for TypeA {
    fn name(&self) -> &'static str { "TypeA" }
}
impl Base for TypeB {
    fn name(&self) -> &'static str { "TypeB" }
}

// Greet — both types
impl Greet for TypeA {
    fn greeting(&self) -> &'static str { "Hello from A" }
}
impl Greet for TypeB {
    fn greeting(&self) -> &'static str { "Hello from B" }
}

// Count — both types
impl Count for TypeA {
    fn count(&self) -> u32 { 42 }
}
impl Count for TypeB {
    fn count(&self) -> u32 { 99 }
}

// Describe — only TypeA
impl Describe for TypeA {
    fn description(&self) -> &'static str { "I am TypeA, the describable" }
}

// ---- downcast function ----

#[inline(never)]
fn check_a(obj: &dyn Base) {
    assert_eq!(obj.name(), "TypeA");

    let greeter = core::cast!(in dyn Base, obj => dyn Greet).unwrap();
    assert_eq!(greeter.greeting(), "Hello from A");

    let counter = core::cast!(in dyn Base, obj => dyn Count).unwrap();
    assert_eq!(counter.count(), 42);

    let describer = core::cast!(in dyn Base, obj => dyn Describe).unwrap();
    assert_eq!(describer.description(), "I am TypeA, the describable");
}

#[inline(never)]
fn check_b(obj: &dyn Base) {
    assert_eq!(obj.name(), "TypeB");

    let greeter = core::cast!(in dyn Base, obj => dyn Greet).unwrap();
    assert_eq!(greeter.greeting(), "Hello from B");

    let counter = core::cast!(in dyn Base, obj => dyn Count).unwrap();
    assert_eq!(counter.count(), 99);

    // Describe is NOT implemented for TypeB — cast must fail.
    assert!(core::cast!(in dyn Base, obj => dyn Describe).is_err());
}

fn main() {
    check_a(&TypeA as &dyn Base);
    check_b(&TypeB as &dyn Base);
}
