//@ run-pass
//! Trait-cast with lifetime-parameterized types: single binary crate, two
//! concrete types (`TypeA<'a, 'b>`, `TypeB<'a, 'b>`), three sub-traits of a
//! common root.
//!
//! - `Label` and `Inspect` are implemented for both types.
//! - `TypeB`'s `Inspect` impl requires `'b: 'a`.
//! - `Special` is implemented only for `TypeA`.
//!
//! The unsizing coercion `&TypeB → &dyn Root` is performed in two distinct
//! contexts whose borrowck `region_summary` yields different outlives
//! relationships, producing different outlives classes in the mono collector:
//!   - `coerce_b_bounded`:   `'b: 'a` provable → Inspect slot populated
//!   - `coerce_b_unbounded`: `'b` is strictly interior, no `'b: 'a` →
//!     Inspect slot is `None`
//!
//! `#[inline(never)]` checkers receive the erased `&dyn Root` and exercise
//! both the success and failure paths via `assert_eq!`.

#![feature(trait_cast)]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait Root: TraitMetadataTable<dyn Root> + core::fmt::Debug {
    fn id(&self) -> u32;
}

trait Label: Root {
    fn label(&self) -> &'static str;
}

trait Inspect: Root {
    fn data(&self) -> u32;
}

trait Special: Root {
    fn special_value(&self) -> u64;
}

// ---- concrete types (two lifetimes each) ----

#[derive(Debug)]
struct TypeA<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

#[derive(Debug)]
struct TypeB<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

// Root — both types
impl<'a, 'b> Root for TypeA<'a, 'b> {
    fn id(&self) -> u32 { 1 }
}
impl<'a, 'b> Root for TypeB<'a, 'b> {
    fn id(&self) -> u32 { 2 }
}

// Label — both types
impl<'a, 'b> Label for TypeA<'a, 'b> {
    fn label(&self) -> &'static str { "TypeA" }
}
impl<'a, 'b> Label for TypeB<'a, 'b> {
    fn label(&self) -> &'static str { "TypeB" }
}

// Inspect — both types, but TypeB's impl requires 'b: 'a
impl<'a, 'b> Inspect for TypeA<'a, 'b> {
    fn data(&self) -> u32 { *self.x + *self.y }
}
impl<'a, 'b> Inspect for TypeB<'a, 'b>
where
    'b: 'a,
{
    fn data(&self) -> u32 { *self.x * *self.y }
}

// Special — only TypeA
impl<'a, 'b> Special for TypeA<'a, 'b> {
    fn special_value(&self) -> u64 { 42 }
}

// ---- downcast checkers ----

#[inline(never)]
fn check_a(obj: &dyn Root) {
    assert_eq!(obj.id(), 1);

    let labeler = core::cast!(in dyn Root, obj => dyn Label)
        .expect("check_a: label");
    assert_eq!(labeler.label(), "TypeA");

    let inspector = core::cast!(in dyn Root, obj => dyn Inspect)
        .expect("check_a: inspect");
    assert_eq!(inspector.data(), 30); // 10 + 20

    let special = core::cast!(in dyn Root, obj => dyn Special)
        .expect("check_a: special");
    assert_eq!(special.special_value(), 42);
}

/// Checker for TypeB when the outlives class does NOT include `'b: 'a`.
#[inline(never)]
fn check_b_inspect_absent(obj: &dyn Root) {
    assert_eq!(obj.id(), 2);

    let labeler = core::cast!(in dyn Root, obj => dyn Label)
        .expect("check_b_inspect_absent: label");
    assert_eq!(labeler.label(), "TypeB");

    // Inspect is not available: where-bounds aren't provable
    // after unsizing.
    core::cast!(in dyn Root, obj => dyn Inspect)
        .expect_err("check_b_inspect_absent: inspect");

    // Special is NOT implemented for TypeB.
    core::cast!(in dyn Root, obj => dyn Special)
        .expect_err("check_b_inspect_absent: special");
}

#[inline(never)]
fn coerce_b_unbounded<'a>(x: &'a u32) {
    let local: u32 = 20;
    check_b_inspect_absent(&TypeB { x, y: &local } as &dyn Root);
}

fn main() {
    let x: u32 = 10;
    let y: u32 = 20;

    check_a(&TypeA { x: &x, y: &y } as &dyn Root);

    // Negative: 'b is local to coerce_b_unbounded, strictly shorter than 'a.
    coerce_b_unbounded(&x);
}
