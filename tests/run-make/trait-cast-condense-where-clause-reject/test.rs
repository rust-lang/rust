//! Condensation with an UNSATISFIABLE RegionOutlives where clause:
//! exercises the explicit-where-clause rejection path in
//! `impl_admissible_under_class`.
//!
//! The `TypeB` impl of `Gated<'a>` carries a `'a: 'static` where
//! clause. `impl_universally_admissible` returns false for it (the
//! RegionOutlives clause disqualifies universal admissibility), so
//! `trait_cast_layout` invokes `condense_outlives_classes` for `Gated`.
//! The `Free` sub-trait's impls are also not universally admissible
//! (shared Self/trait params), so Free also goes through condensation.
//!
//! All call contexts use scoped (non-'static) lifetimes, so the only
//! materialized outlives class is `{empty}`. Under that class:
//!   - TypeA::Gated admits  (no where clause)
//!   - TypeB::Gated REJECTS (the 'a: 'static bound never holds because
//!     reachability has no 'a→'static edge)
//! Matrix row for Gated: `[TypeA: ✓, TypeB: ✗]`.
//!
//! With one row per sub-trait, condensation produces exactly one slot
//! per sub-trait. Total: 2 slots. This pins the explicit-where-clause
//! rejection path — if `impl_admissible_under_class` accidentally
//! started admitting `'a: 'static` under a scoped class, the slot
//! would still be 1 for Gated (row pattern would just be `[✓, ✓]`),
//! so runtime behavior verification (that the TypeB cast rejects) is
//! what would catch that regression.
//!
//! Expected resolutions in the post-mono MIR:
//!   trait_metadata_table_len<dyn Root<'_>>()  → 2_usize
//!   trait_metadata_index                        → 0_usize, 1_usize

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![allow(dead_code, unused_variables)]
#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait Root<'a>: TraitMetadataTable<dyn Root<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}

trait Free<'a>: Root<'a> {
    fn free_val(&self) -> u32;
}

trait Gated<'a>: Root<'a> {
    fn gated_val(&self) -> u32;
}

// ---- concrete types ----

#[derive(Debug)]
struct TypeA<'a> {
    x: &'a u32,
}

#[derive(Debug)]
struct TypeB<'a> {
    x: &'a u32,
}

// Universal impls — both types, Root + Free.
impl<'a> Root<'a> for TypeA<'a> {
    fn id(&self) -> u32 {
        1
    }
}
impl<'a> Root<'a> for TypeB<'a> {
    fn id(&self) -> u32 {
        2
    }
}
impl<'a> Free<'a> for TypeA<'a> {
    fn free_val(&self) -> u32 {
        10
    }
}
impl<'a> Free<'a> for TypeB<'a> {
    fn free_val(&self) -> u32 {
        20
    }
}

// Gated — TypeA is universal, TypeB has a where clause that is never
// provable for a scoped (non-'static) 'a.
impl<'a> Gated<'a> for TypeA<'a> {
    fn gated_val(&self) -> u32 {
        *self.x
    }
}
impl<'a> Gated<'a> for TypeB<'a>
where
    'a: 'static,
{
    fn gated_val(&self) -> u32 {
        *self.x + 100
    }
}

// ---- multiple coercion/cast contexts (all with scoped 'a) ----

#[inline(never)]
fn ctx_simple<'a>(x: &'a u32) {
    let obj: &dyn Root<'a> = &TypeA { x };
    let _ = core::cast!(in dyn Root<'a>, obj => dyn Free<'a>);
    let _ = core::cast!(in dyn Root<'a>, obj => dyn Gated<'a>);
}

#[inline(never)]
fn ctx_interior<'a>(x: &'a u32) {
    let local: u32 = 7;
    let obj: &dyn Root<'_> = &TypeB { x: &local };
    let _ = core::cast!(in dyn Root<'_>, obj => dyn Free<'_>);
    let _ = core::cast!(in dyn Root<'_>, obj => dyn Gated<'_>);
    let _ = x;
}

#[inline(never)]
fn ctx_outer<'a, 'b>(x: &'a u32, y: &'b u32)
where
    'a: 'b,
{
    let obj: &dyn Root<'b> = &TypeA { x: y };
    let _ = core::cast!(in dyn Root<'b>, obj => dyn Free<'b>);
    let _ = core::cast!(in dyn Root<'b>, obj => dyn Gated<'b>);
    let _ = x;
}

#[inline(never)]
fn ctx_both_b<'a>(x: &'a u32) {
    let inner: u32 = 99;
    let obj_a: &dyn Root<'_> = &TypeA { x };
    let obj_b: &dyn Root<'_> = &TypeB { x: &inner };
    let _ = core::cast!(in dyn Root<'_>, obj_a => dyn Gated<'_>);
    let _ = core::cast!(in dyn Root<'_>, obj_b => dyn Gated<'_>);
}

fn main() {
    let a: u32 = 3;
    let b: u32 = 5;
    ctx_simple(&a);
    ctx_interior(&a);
    ctx_outer(&a, &b);
    ctx_both_b(&a);
}
