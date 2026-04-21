//! Condensation with PARAMETER ALIASING in an impl: exercises the
//! "one impl param maps to multiple distinct bvs" rule in
//! `impl_admissible_under_class`.
//!
//! The dyn type is `dyn Aliased<'a, 'b>` (2 bound variables).
//! `TypeB` is implemented only at `Aliased<'c, 'c>` — one impl param
//! `'c` maps to BOTH bvs 0 and 1. The aliasing rule says: when one
//! impl param maps to multiple distinct bvs, those bvs must be
//! MUTUALLY OUTLIVES under the current outlives class. With an empty
//! class, the mutual-outlives check fails, so
//! `impl_admissible_under_class` rejects the `TypeB::Aliased` impl.
//!
//! Param aliasing ALSO disqualifies the impl from
//! `impl_universally_admissible`, so the fast path is bypassed and
//! `condense_outlives_classes` is invoked for `Aliased`. The
//! admissibility matrix row for the single empty class is
//! `[TypeA: ✓, TypeB: ✗]` — one row → one condensed slot.
//!
//! The `Free<'a, 'b>` sub-trait has no where clauses and no param
//! aliasing but still goes through condensation because of shared
//! Self/trait params (Self-anchored-param rule). Both sub-traits
//! therefore receive one slot each via condensation.
//!
//! Expected total table length: 2 slots (1 Free + 1 Aliased).

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![allow(dead_code, unused_variables)]
#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait Root<'a, 'b>: TraitMetadataTable<dyn Root<'a, 'b>> + core::fmt::Debug {
    fn id(&self) -> u32;
}

trait Free<'a, 'b>: Root<'a, 'b> {
    fn free_val(&self) -> u32;
}

trait Aliased<'a, 'b>: Root<'a, 'b> {
    fn aliased_val(&self) -> u32;
}

// ---- concrete types ----

#[derive(Debug)]
struct TypeA<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

/// `TypeB` carries a single lifetime parameter that is used to
/// instantiate both bvs of `Aliased<'c, 'c>` below — the aliasing
/// pattern.
#[derive(Debug)]
struct TypeB<'c> {
    v: &'c u32,
}

// Universal-looking impls (still on the condensation path because of
// shared Self/trait params).
impl<'a, 'b> Root<'a, 'b> for TypeA<'a, 'b> {
    fn id(&self) -> u32 {
        1
    }
}
impl<'c> Root<'c, 'c> for TypeB<'c> {
    fn id(&self) -> u32 {
        2
    }
}
impl<'a, 'b> Free<'a, 'b> for TypeA<'a, 'b> {
    fn free_val(&self) -> u32 {
        10
    }
}
impl<'c> Free<'c, 'c> for TypeB<'c> {
    fn free_val(&self) -> u32 {
        20
    }
}

// Aliased — TypeA impls without aliasing; TypeB impls with 'c used
// for BOTH positions of the trait ref (one param → two bvs).
impl<'a, 'b> Aliased<'a, 'b> for TypeA<'a, 'b> {
    fn aliased_val(&self) -> u32 {
        *self.x + *self.y
    }
}
impl<'c> Aliased<'c, 'c> for TypeB<'c> {
    fn aliased_val(&self) -> u32 {
        *self.v
    }
}

// ---- call contexts ----

#[inline(never)]
fn ctx_a<'a>(x: &'a u32) {
    let local: u32 = 7;
    let obj: &dyn Root<'_, '_> = &TypeA { x, y: &local };
    let _ = core::cast!(in dyn Root<'_, '_>, obj => dyn Free<'_, '_>);
    let _ = core::cast!(in dyn Root<'_, '_>, obj => dyn Aliased<'_, '_>);
}

#[inline(never)]
fn ctx_b<'c>(v: &'c u32) {
    // Single lifetime — both slots instantiated from 'c.
    let obj: &dyn Root<'c, 'c> = &TypeB { v };
    let _ = core::cast!(in dyn Root<'c, 'c>, obj => dyn Free<'c, 'c>);
    let _ = core::cast!(in dyn Root<'c, 'c>, obj => dyn Aliased<'c, 'c>);
}

fn main() {
    let a: u32 = 3;
    ctx_a(&a);
    ctx_b(&a);
}
