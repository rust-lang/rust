//@ run-pass
//! Erasure-safety outlives-based correspondence tests.
//!
//! Given that the structural surjectivity check passes, these tests
//! verify that the outlives entries at the coercion site correctly
//! determine erasure safety. The key rule: for each (target_bv,
//! root_bv) pair in the structural mapping, mutual outlives (both
//! directions) must hold. When a target bv maps to multiple root bvs,
//! those root bvs must also be mutually equivalent.
//!
//! Each test case controls the outlives environment at the coercion
//! site by varying whether lifetime parameters have provable outlives
//! relationships.

#![feature(trait_cast)]

#![crate_type = "bin"]

#![allow(dead_code, unused_variables)]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Case 1: t0 -> r0, n=1, mutual outlives {(0,1),(1,0)} => true
//
// Simple one-to-one mapping with both outlives directions present.
// This is the positive case: the coercion site has evidence that
// the root bv and target bv refer to the same lifetime.
// =========================================================================

trait Root1<'a>: TraitMetadataTable<dyn Root1<'a>> + core::fmt::Debug {
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Target1<'a>: Root1<'a> {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct T1<'a> { _x: &'a u32 }

impl<'a> Root1<'a> for T1<'a> {
    fn id(&self) -> u32 { 1 }
}
impl<'a> Target1<'a> for T1<'a> {
    fn val(&self) -> u32 { 100 }
}

/// Both root and target share the same universal lifetime 'a.
/// The mono collector produces mutual outlives entries.
#[inline(never)]
fn case1_mutual<'a>(x: &'a u32) {
    let obj: &dyn Root1<'a> = &T1 { _x: x };
    let target = core::cast!(in dyn Root1<'_>, obj => dyn Target1<'_>).expect("case1_mutual");
    assert_eq!(target.val(), 100);
}

// =========================================================================
// Case 2: Asymmetric outlives — cast must fail
//
// The impl requires 'a: 'b, so the table slot is only populated when
// that holds. Even when the impl is admissible, erasure safety requires
// the dyn bv for the root and target to be provably equivalent. Here
// the coercion site lacks that evidence, so the cast must fail.
// =========================================================================

trait Root2<'a>: TraitMetadataTable<dyn Root2<'a>> + core::fmt::Debug {
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Target2<'a>: Root2<'a> {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct T2<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Root2<'a> for T2<'a, 'b> {
    fn id(&self) -> u32 { 2 }
}
// The impl requires 'a: 'b, so the table slot is only populated when
// that holds. Additionally, erasure safety requires the dyn bv for the
// root and target to be provably equivalent.
impl<'a, 'b> Target2<'a> for T2<'a, 'b>
where
    'a: 'b,
{
    fn val(&self) -> u32 { 200 }
}

/// We can't prove anything about a hidden lifetime. The cast must fail.
#[inline(never)]
fn case2_both_directions<'a>(x: &'a u32) {
    let obj: &dyn Root2<'_> = &T2 { _x: x, _y: x };
    core::cast!(in dyn Root2<'_>, obj => dyn Target2<'_>).expect_err("case2_both_directions");
}

/// We can't prove anything about a hidden lifetime. The cast must fail.
#[inline(never)]
fn case2_no_evidence<'a>(x: &'a u32) {
    let local: u32 = 99;
    let obj: &dyn Root2<'_> = &T2 { _x: &local, _y: x };
    core::cast!(in dyn Root2<'_>, obj => dyn Target2<'_>).expect_err("case2_no_evidence");
}

// =========================================================================
// Case 3: Swap mapping t0 -> r1, t1 -> r0, n=2
//         with mutual outlives {(1,2),(2,1),(0,3),(3,0)} => true
//
// Both pairs (t0<->r1) and (t1<->r0) have mutual outlives in the
// combined index space.
// =========================================================================

trait Root3<'a, 'b>: TraitMetadataTable<dyn Root3<'a, 'b>> + core::fmt::Debug {
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Target3<'a, 'b>: Root3<'b, 'a> {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct T3<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Root3<'a, 'b> for T3<'a, 'b> {
    fn id(&self) -> u32 { 3 }
}
impl<'a, 'b> Target3<'a, 'b> for T3<'b, 'a> {
    fn val(&self) -> u32 { 300 }
}

/// When both lifetimes are the same, the swap is trivially satisfied.
/// Explicit 'a at the coercion site provides invariant constraints;
/// the cast site uses inferred lifetimes.
#[inline(never)]
fn case3_swap_bounded<'a>(x: &'a u32) {
    let obj: &dyn Root3<'a, 'a> = &T3 { _x: x, _y: x };
    let target = core::cast!(in dyn Root3<'_, '_>, obj => dyn Target3<'_, '_>)
        .expect("case3_swap_bounded");
    assert_eq!(target.val(), 300);
}

/// When lifetimes are unrelated, the swap mapping still succeeds
/// because each structural pair maps to the same concrete lifetime:
/// target_bv0 and root_bv1 both hold 'local, target_bv1 and root_bv0
/// both hold 'a. The resulting reference is bounded by min('a, 'local).
#[inline(never)]
fn case3_swap_unbounded<'a>(x: &'a u32) {
    let local: u32 = 99;
    let obj: &(dyn Root3<'a, '_> + '_) = &T3 { _x: x, _y: &local };
    let target = core::cast!(in dyn Root3<'_, '_>, obj => dyn Target3<'_, '_>)
        .expect("case3_swap_unbounded");
    assert_eq!(target.val(), 300);
}

// =========================================================================
// Case 4: Fan-out t0 -> {r0, r1}, n=2
//         with ALL mutual: {(0,2),(2,0),(1,2),(2,1),(0,1),(1,0)} => true
//
// Target has one bv mapping to both root bvs. All pairs must be
// mutually equivalent.
// =========================================================================

trait Root4<'a, 'b>: TraitMetadataTable<dyn Root4<'a, 'b>> + core::fmt::Debug {
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Target4<'a>: Root4<'a, 'a> {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct T4<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Root4<'a, 'b> for T4<'a, 'b> {
    fn id(&self) -> u32 { 4 }
}
impl<'a> Target4<'a> for T4<'a, 'a> {
    fn val(&self) -> u32 { 400 }
}

/// When 'a == 'b (same reference), all three pairs (t0<->r0, t0<->r1,
/// r0<->r1) are trivially mutual.
/// Explicit 'a at the coercion site; cast lifetimes inferred.
#[inline(never)]
fn case4_all_mutual<'a>(x: &'a u32) {
    let obj: &dyn Root4<'a, 'a> = &T4 { _x: x, _y: x };
    let target = core::cast!(in dyn Root4<'_, '_>, obj => dyn Target4<'_>)
        .expect("case4_all_mutual");
    assert_eq!(target.val(), 400);
}

// When 'a and 'b are unrelated r0 and r1 are not equivalent, so the
// fan-out mapping fails even when the target bv outlives each root bv
// individually. Borrowck rules out a direct in-source reproduction.

fn main() {
    let x: u32 = 42;

    case1_mutual(&x);

    case2_both_directions(&x);
    case2_no_evidence(&x);

    case3_swap_bounded(&x);
    case3_swap_unbounded(&x);

    case4_all_mutual(&x);
}
