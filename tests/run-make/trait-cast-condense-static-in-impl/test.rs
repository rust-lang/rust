//! Condensation with a concrete `'static` lifetime in an impl's
//! trait ref: exercises the `ReStatic`-in-trait-arg branch of
//! `impl_admissible_under_class`.
//!
//! `TypeB` is implemented only at `Anchored<'static>` — the impl's
//! trait-ref arg at position 0 is the literal `'static`, not a free
//! param. The rule then requires that for every bv at that position,
//! the outlives class implies `bv outlives 'static`. Under the only
//! materialized (empty) outlives class, reachability contains no
//! `bv → 'static` edge, so the admissibility check rejects the
//! TypeB::Anchored impl.
//!
//! Concrete `'static` in the trait ref also disqualifies
//! `impl_universally_admissible` (condition: "no concrete lifetimes in
//! the trait ref"), so `trait_cast_layout` bypasses the fast path and
//! invokes `condense_outlives_classes` for `Anchored`.
//! Matrix row for Anchored (empty class): `[TypeA: ✓, TypeB: ✗]` →
//! one row → one condensed slot.
//!
//! `Free<'a>` has no concrete lifetimes but still goes through
//! condensation via shared Self/trait params. Total: 2 slots.

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

trait Anchored<'a>: Root<'a> {
    fn anchored_val(&self) -> u32;
}

// ---- concrete types ----

#[derive(Debug)]
struct TypeA<'a> {
    x: &'a u32,
}

/// `TypeB` uses `'static` for its `Root`/`Anchored` impls, so the
/// admissibility check sees `ReStatic` at the trait-ref lifetime
/// position.
#[derive(Debug)]
struct TypeB {
    v: &'static u32,
}

impl<'a> Root<'a> for TypeA<'a> {
    fn id(&self) -> u32 {
        1
    }
}
impl Root<'static> for TypeB {
    fn id(&self) -> u32 {
        2
    }
}
impl<'a> Free<'a> for TypeA<'a> {
    fn free_val(&self) -> u32 {
        10
    }
}
impl Free<'static> for TypeB {
    fn free_val(&self) -> u32 {
        20
    }
}

impl<'a> Anchored<'a> for TypeA<'a> {
    fn anchored_val(&self) -> u32 {
        *self.x
    }
}
impl Anchored<'static> for TypeB {
    fn anchored_val(&self) -> u32 {
        *self.v
    }
}

// ---- call contexts ----

#[inline(never)]
fn ctx_scoped<'a>(x: &'a u32) {
    let obj: &dyn Root<'a> = &TypeA { x };
    let _ = core::cast!(in dyn Root<'a>, obj => dyn Free<'a>);
    let _ = core::cast!(in dyn Root<'a>, obj => dyn Anchored<'a>);
}

#[inline(never)]
fn ctx_static() {
    static X: u32 = 42;
    let obj_a: &dyn Root<'static> = &TypeA { x: &X };
    let _ = core::cast!(in dyn Root<'static>, obj_a => dyn Free<'static>);
    let _ = core::cast!(in dyn Root<'static>, obj_a => dyn Anchored<'static>);

    let obj_b: &dyn Root<'static> = &TypeB { v: &X };
    let _ = core::cast!(in dyn Root<'static>, obj_b => dyn Free<'static>);
    let _ = core::cast!(in dyn Root<'static>, obj_b => dyn Anchored<'static>);
}

fn main() {
    let a: u32 = 3;
    ctx_scoped(&a);
    ctx_static();
}
