//@ run-pass
//! Trait-cast torture tests: complex lifetime and hierarchy scenarios.
//!
//! Stress tests combining multiple trait-cast features to exercise corner
//! cases in the mono collector, erasure-safety checks, and metadata table
//! generation.
//!
//! Test cases:
//!   1. Wide hierarchy: root with six sub-traits, two concrete types with
//!      different impl coverage.
//!   2. Three lifetimes: full merge (3 -> 1) vs. partial merge (3 -> 2).
//!   3. Two independent root hierarchies on the same concrete type.
//!   4. Zero-sized type with PhantomData lifetime.
//!   5. Deep chain with where-clause gating at the intermediate level.

#![feature(trait_cast)]
#![allow(dead_code, unused_variables)]

#![crate_type = "bin"]

extern crate core;
use core::marker::{PhantomData, TraitMetadataTable};

// =========================================================================
// Case 1: Wide hierarchy — root with six sub-traits
//
// TypeAll implements all six sub-traits.
// TypeSparse implements only W1, W3, and W5 — the even-numbered
// sub-traits are absent.  Tests that a wide metadata table correctly
// interleaves populated and empty slots.
// =========================================================================

trait WRoot<'a>: TraitMetadataTable<dyn WRoot<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}

trait W1<'a>: WRoot<'a> { fn w1(&self) -> u32; }
trait W2<'a>: WRoot<'a> { fn w2(&self) -> u32; }
trait W3<'a>: WRoot<'a> { fn w3(&self) -> u32; }
trait W4<'a>: WRoot<'a> { fn w4(&self) -> u32; }
trait W5<'a>: WRoot<'a> { fn w5(&self) -> u32; }
trait W6<'a>: WRoot<'a> { fn w6(&self) -> u32; }

#[derive(Debug)]
struct TypeAll<'a> { _x: &'a u32 }

#[derive(Debug)]
struct TypeSparse<'a> { _x: &'a u32 }

// TypeAll — everything
impl<'a> WRoot<'a> for TypeAll<'a> { fn id(&self) -> u32 { 1 } }
impl<'a> W1<'a> for TypeAll<'a> { fn w1(&self) -> u32 { 10 } }
impl<'a> W2<'a> for TypeAll<'a> { fn w2(&self) -> u32 { 20 } }
impl<'a> W3<'a> for TypeAll<'a> { fn w3(&self) -> u32 { 30 } }
impl<'a> W4<'a> for TypeAll<'a> { fn w4(&self) -> u32 { 40 } }
impl<'a> W5<'a> for TypeAll<'a> { fn w5(&self) -> u32 { 50 } }
impl<'a> W6<'a> for TypeAll<'a> { fn w6(&self) -> u32 { 60 } }

// TypeSparse — odd-numbered only
impl<'a> WRoot<'a> for TypeSparse<'a> { fn id(&self) -> u32 { 2 } }
impl<'a> W1<'a> for TypeSparse<'a> { fn w1(&self) -> u32 { 11 } }
impl<'a> W3<'a> for TypeSparse<'a> { fn w3(&self) -> u32 { 31 } }
impl<'a> W5<'a> for TypeSparse<'a> { fn w5(&self) -> u32 { 51 } }

/// TypeAll: all six sub-traits available.
#[inline(never)]
fn case1_all<'a>(x: &'a u32) {
    let obj: &dyn WRoot<'_> = &TypeAll { _x: x };
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W1<'_>).expect("W1").w1(), 10);
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W2<'_>).expect("W2").w2(), 20);
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W3<'_>).expect("W3").w3(), 30);
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W4<'_>).expect("W4").w4(), 40);
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W5<'_>).expect("W5").w5(), 50);
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W6<'_>).expect("W6").w6(), 60);
}

/// TypeSparse: odd sub-traits available, even absent.
#[inline(never)]
fn case1_sparse<'a>(x: &'a u32) {
    let obj: &dyn WRoot<'_> = &TypeSparse { _x: x };
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W1<'_>).expect("W1").w1(), 11);
    core::cast!(in dyn WRoot<'_>, obj => dyn W2<'_>).expect_err("W2");
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W3<'_>).expect("W3").w3(), 31);
    core::cast!(in dyn WRoot<'_>, obj => dyn W4<'_>).expect_err("W4");
    assert_eq!(core::cast!(in dyn WRoot<'_>, obj => dyn W5<'_>).expect("W5").w5(), 51);
    core::cast!(in dyn WRoot<'_>, obj => dyn W6<'_>).expect_err("W6");
}

// =========================================================================
// Case 2: Three lifetimes — full merge vs. partial merge
//
// Tri<'a, 'b, 'c> has three binder variables.  TriMerge<'a> merges
// all three (fan-out t0 -> {r0, r1, r2}), requiring pairwise mutual
// equivalence.  TriPartial<'a, 'c> merges only the first two
// (fan-out t0 -> {r0, r1}, plus identity t1 -> r2).
// =========================================================================

trait Tri<'a, 'b, 'c>: TraitMetadataTable<dyn Tri<'a, 'b, 'c>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait TriMerge<'a>: Tri<'a, 'a, 'a> {
    fn merged(&self) -> u32;
}
trait TriPartial<'a, 'c>: Tri<'a, 'a, 'c> {
    fn partial(&self) -> u32;
}

#[derive(Debug)]
struct Triple<'a, 'b, 'c> {
    _x: &'a u32,
    _y: &'b u32,
    _z: &'c u32,
}

impl<'a, 'b, 'c> Tri<'a, 'b, 'c> for Triple<'a, 'b, 'c> {
    fn id(&self) -> u32 { 3 }
}
impl<'a> TriMerge<'a> for Triple<'a, 'a, 'a> {
    fn merged(&self) -> u32 { 333 }
}
impl<'a, 'c> TriPartial<'a, 'c> for Triple<'a, 'a, 'c> {
    fn partial(&self) -> u32 { 334 }
}

/// All three lifetimes identical => both full and partial merge succeed.
/// (Borrowck forces binder variables toward equivalence via the
/// TraitMetadataTable marker, so the negative fan-out cases — e.g.,
/// first two same but third different — are not independently testable.)
#[inline(never)]
fn case2_all_same<'a>(x: &'a u32) {
    let obj: &dyn Tri<'_, '_, '_> = &Triple { _x: x, _y: x, _z: x };
    let m = core::cast!(in dyn Tri<'_, '_, '_>, obj => dyn TriMerge<'_>)
        .expect("all_same: merge");
    assert_eq!(m.merged(), 333);
    let p = core::cast!(in dyn Tri<'_, '_, '_>, obj => dyn TriPartial<'_, '_>)
        .expect("all_same: partial");
    assert_eq!(p.partial(), 334);
}

// =========================================================================
// Case 3: Two independent root hierarchies on the same concrete type
//
// MultiRoot<'a> implements both RootA (no lifetime params) and
// RootB<'a> (with lifetime param).  Each root has its own sub-trait.
// Casts through each hierarchy are independent — the per-root
// metadata tables are generated separately.
// =========================================================================

trait RootA: TraitMetadataTable<dyn RootA> + core::fmt::Debug {
    fn id_a(&self) -> u32;
}
trait SubA: RootA {
    fn val_a(&self) -> u32;
}
trait SubA2: RootA {
    fn val_a2(&self) -> u32;
}

trait RootB<'a>: TraitMetadataTable<dyn RootB<'a>> + core::fmt::Debug {
    fn id_b(&self) -> u32;
}
trait SubB<'a>: RootB<'a> {
    fn val_b(&self) -> u32;
}

#[derive(Debug)]
struct MultiRoot<'a> { _x: &'a u32 }

impl<'a> RootA for MultiRoot<'a> {
    fn id_a(&self) -> u32 { 1 }
}
impl<'a> SubA for MultiRoot<'a> {
    fn val_a(&self) -> u32 { 10 }
}
impl<'a> SubA2 for MultiRoot<'a> {
    fn val_a2(&self) -> u32 { 15 }
}
impl<'a> RootB<'a> for MultiRoot<'a> {
    fn id_b(&self) -> u32 { 2 }
}
impl<'a> SubB<'a> for MultiRoot<'a> {
    fn val_b(&self) -> u32 { 20 }
}

/// The same concrete value is independently usable through both roots.
#[inline(never)]
fn case3_dual_roots<'a>(x: &'a u32) {
    let val = MultiRoot { _x: x };

    // RootA hierarchy (no lifetimes).
    let obj_a: &dyn RootA = &val;
    let sub_a = core::cast!(in dyn RootA, obj_a => dyn SubA).expect("RootA -> SubA");
    assert_eq!(sub_a.val_a(), 10);
    let sub_a2 = core::cast!(in dyn RootA, obj_a => dyn SubA2).expect("RootA -> SubA2");
    assert_eq!(sub_a2.val_a2(), 15);

    // RootB hierarchy (with lifetime).
    let obj_b: &dyn RootB<'_> = &val;
    let sub_b = core::cast!(in dyn RootB<'_>, obj_b => dyn SubB<'_>).expect("RootB -> SubB");
    assert_eq!(sub_b.val_b(), 20);
}

// =========================================================================
// Case 4: Zero-sized type with PhantomData lifetime
//
// A ZST (zero-sized type) that carries a lifetime only through
// PhantomData.  The data pointer in the trait object is dangling
// (aligned to ZST conventions).  Tests that the trait-cast machinery
// handles ZSTs correctly.
// =========================================================================

trait RootZ<'a>: TraitMetadataTable<dyn RootZ<'a>> + core::fmt::Debug {
    fn tag(&self) -> u32;
}
trait SubZ<'a>: RootZ<'a> {
    fn ztag(&self) -> u32;
}

#[derive(Debug)]
struct Zst<'a> {
    _marker: PhantomData<&'a ()>,
}

impl<'a> RootZ<'a> for Zst<'a> {
    fn tag(&self) -> u32 { 0 }
}
impl<'a> SubZ<'a> for Zst<'a> {
    fn ztag(&self) -> u32 { 1 }
}

#[inline(never)]
fn case4_zst<'a>(_anchor: &'a u32) {
    let z: Zst<'a> = Zst { _marker: PhantomData };
    let obj: &dyn RootZ<'_> = &z;
    assert_eq!(obj.tag(), 0);
    let sub = core::cast!(in dyn RootZ<'_>, obj => dyn SubZ<'_>).expect("zst");
    assert_eq!(sub.ztag(), 1);
}

// =========================================================================
// Case 5: Deep chain with where-clause gating (hidden lifetimes)
//
// Sub5: Mid5: Root5, where Root5 has NO lifetime params (both
// lifetimes are hidden).  Mid5 and Sub5 impls require 'b: 'a.
// Tests that where-clause gating propagates through a chain.
// Uses the proven hidden-lifetime pattern from lifetime-bounded-downcast.
// =========================================================================

trait Root5: TraitMetadataTable<dyn Root5> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Mid5: Root5 {
    fn mid(&self) -> u32;
}
trait Sub5: Mid5 {
    fn sub(&self) -> u32;
}

#[derive(Debug)]
struct Deep<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Root5 for Deep<'a, 'b> {
    fn id(&self) -> u32 { 5 }
}
impl<'a, 'b> Mid5 for Deep<'a, 'b>
where
    'b: 'a,
{
    fn mid(&self) -> u32 { 51 }
}
impl<'a, 'b> Sub5 for Deep<'a, 'b>
where
    'b: 'a,
{
    fn sub(&self) -> u32 { 52 }
}

/// 'b is a local lifetime, strictly shorter than 'a.
/// 'b: 'a does not hold => neither Mid5 nor Sub5 is available.
#[inline(never)]
fn case5_unbounded<'a>(x: &'a u32) {
    let local: u32 = 99;
    let obj: &dyn Root5 = &Deep { _x: x, _y: &local };
    core::cast!(in dyn Root5, obj => dyn Mid5).expect_err("mid");
    core::cast!(in dyn Root5, obj => dyn Sub5).expect_err("sub");
}

fn main() {
    let x: u32 = 42;

    eprintln!("=== case1: wide hierarchy ===");
    case1_all(&x);
    case1_sparse(&x);

    eprintln!("=== case2: three lifetimes ===");
    case2_all_same(&x);

    eprintln!("=== case3: dual roots ===");
    case3_dual_roots(&x);

    eprintln!("=== case4: ZST ===");
    case4_zst(&x);

    eprintln!("=== case5: deep chain ===");
    case5_unbounded(&x);
}
