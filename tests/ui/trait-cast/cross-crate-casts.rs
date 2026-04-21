//@ run-pass
//@ aux-build:cross_crate_lib.rs
//! Cross-crate trait-cast validation.
//!
//! The upstream rlib `cross_crate_lib` defines a four-trait graph
//! (`Root`/`Greet`/`Count`/`Describe`) and two concrete types:
//!
//!   * `LibTypeA` — impls every sub-trait.
//!   * `LibTypeB` — impls `Greet` and `Count` but **not** `Describe`.
//!
//! The downstream bin (this crate, a global crate by virtue of being an
//! executable) performs casts through three distinct channels to exercise
//! the cross-crate `DelayedInstance` / `trait_cast_intrinsics` rmeta
//! pipeline end-to-end:
//!
//!   1. Direct cast of a stack-allocated library type:
//!      `&LibTypeA as &dyn Root => dyn Greet/Count/Describe`.
//!      The cast site is in the bin; the library's vtable for
//!      `LibTypeA` was built upstream.
//!   2. Cast against a `Box<dyn Root>` returned from the library.  The
//!      `LibTypeA -> dyn Root` unsizing happened upstream, so the
//!      `TraitMetadataTable::derived_metadata_table` entry in the
//!      vtable is a monomorphization of the upstream intrinsic that
//!      must have been decoded from the library's rmeta.
//!   3. Cast via `cross_crate_lib::try_describe_from_lib`, a *generic*
//!      `&dyn Root -> Option<&'static str>` function defined upstream
//!      whose body contains a `core::cast!` expansion.  The upstream
//!      crate records this as a `DelayedInstance`, and the global phase
//!      in the bin consumes `delayed_codegen_requests(upstream)` to
//!      splice in the augmented callee.
//!
//! The test additionally validates that a sub-trait defined *downstream*
//! (`BinExtra`) can extend the upstream `Root` and participate in the
//! graph.  Library types (which have no `BinExtra` impl) must fail that
//! cast; a bin-local type that does impl `BinExtra` must succeed.

#![feature(trait_cast)]

#![crate_type = "bin"]

extern crate core;
extern crate cross_crate_lib;

use cross_crate_lib::{
    Count, Describe, Greet, LibTypeA, LibTypeB, Root, lib_boxed_a, lib_boxed_b,
    try_describe_from_lib,
};

// ---- downstream sub-trait extending an upstream root ----

trait BinExtra: Root {
    fn extra(&self) -> &'static str;
}

#[derive(Debug)]
struct BinType;

impl Root for BinType {
    fn name(&self) -> &'static str { "BinType" }
}
impl BinExtra for BinType {
    fn extra(&self) -> &'static str { "downstream only" }
}

// ---- check functions (one per concrete type) ----
//
// `#[inline(never)]` keeps the cast sites in distinct MIR bodies so the
// per-instance `call_id` chain the collector threads through
// `DelayedInstance::callee_substitutions` is exercised.

#[inline(never)]
fn check_lib_type_a(obj: &dyn Root) {
    assert_eq!(obj.name(), "LibTypeA");

    let g = core::cast!(in dyn Root, obj => dyn Greet).expect("A: Greet");
    assert_eq!(g.greeting(), "hello from LibTypeA");

    let c = core::cast!(in dyn Root, obj => dyn Count).expect("A: Count");
    assert_eq!(c.count(), 1);

    let d = core::cast!(in dyn Root, obj => dyn Describe).expect("A: Describe");
    assert_eq!(d.description(), "the describable one");

    // Downstream-defined sub-trait: LibTypeA has no BinExtra impl.
    assert!(core::cast!(in dyn Root, obj => dyn BinExtra).is_err());
}

#[inline(never)]
fn check_lib_type_b(obj: &dyn Root) {
    assert_eq!(obj.name(), "LibTypeB");

    let g = core::cast!(in dyn Root, obj => dyn Greet).expect("B: Greet");
    assert_eq!(g.greeting(), "hello from LibTypeB");

    let c = core::cast!(in dyn Root, obj => dyn Count).expect("B: Count");
    assert_eq!(c.count(), 2);

    // LibTypeB has no Describe impl upstream.
    assert!(core::cast!(in dyn Root, obj => dyn Describe).is_err());
    // ...nor the downstream BinExtra.
    assert!(core::cast!(in dyn Root, obj => dyn BinExtra).is_err());
}

#[inline(never)]
fn check_bin_type(obj: &dyn Root) {
    assert_eq!(obj.name(), "BinType");

    // BinType implements only Root + BinExtra locally — the three
    // upstream sub-traits must all fail.
    assert!(core::cast!(in dyn Root, obj => dyn Greet).is_err());
    assert!(core::cast!(in dyn Root, obj => dyn Count).is_err());
    assert!(core::cast!(in dyn Root, obj => dyn Describe).is_err());

    let e = core::cast!(in dyn Root, obj => dyn BinExtra).expect("Bin: BinExtra");
    assert_eq!(e.extra(), "downstream only");
}

fn main() {
    // (1) direct downstream cast, lib type on the stack.
    check_lib_type_a(&LibTypeA as &dyn Root);
    check_lib_type_b(&LibTypeB as &dyn Root);

    // (2) library-allocated Box<dyn Root>: vtable was built upstream.
    check_lib_type_a(&*lib_boxed_a());
    check_lib_type_b(&*lib_boxed_b());

    // (3) upstream cast site instantiated downstream.
    assert_eq!(try_describe_from_lib(&LibTypeA), Some("the describable one"));
    assert_eq!(try_describe_from_lib(&LibTypeB), None);
    // BinType flows through the upstream cast site but has no Describe
    // impl — the global phase still needs to emit a valid table entry
    // (None) for it under the upstream `Describe` slot.
    assert_eq!(try_describe_from_lib(&BinType), None);

    // Downstream concrete type casting to a downstream sub-trait.
    check_bin_type(&BinType as &dyn Root);
}
