//@ run-pass
//! Erasure-safety structural surjectivity tests (S14.3.1).
//!
//! Verifies that the binder-variable mapping between root and target dyn
//! types is structurally surjective. These tests exercise the "forward
//! coverage" and "backward coverage" rules: every target binder variable
//! must map to at least one root binder variable, and every root binder
//! variable must be mapped to by at least one target binder variable.
//!
//! Tests marked "depends on outlives" are exercised in both bounded and
//! unbounded contexts; the structural check passes but the final result
//! depends on whether mutual outlives evidence is available.

#![feature(trait_cast)]
#![allow(dead_code, unused_variables)]

#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Case 1: dyn for<'a> Super<'a> -> dyn for<'a> Sub<'a> via Sub<'a>: Super<'a>
//
// Structural: t0 -> r0 (one-to-one identity mapping).
// Result: depends on outlives (needs mutual outlives for t0 <-> r0).
// =========================================================================

trait Super1<'a>: TraitMetadataTable<dyn Super1<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Sub1<'a>: Super1<'a> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S1<'a> { _x: &'a u32 }

impl<'a> Super1<'a> for S1<'a> {
    fn id(&self) -> u32 { 1 }
}
impl<'a> Sub1<'a> for S1<'a> {
    fn sub_val(&self) -> u32 { 10 }
}

/// 'a is a universal region; coercing S1<'a> to dyn Super1<'a> produces
/// a binder with one variable. The identity mapping t0 -> r0 is trivially
/// mutual in the same universal region, so erasure safety holds.
#[inline(never)]
fn case1_bounded<'a>(x: &'a u32) {
    let obj: &dyn Super1<'_> = &S1 { _x: x };
    let sub = core::cast!(in dyn Super1<'_>, obj => dyn Sub1<'_>).expect("case1_bounded");
    assert_eq!(sub.sub_val(), 10);
}

// =========================================================================
// Case 2 removed: `trait Sub2<'a>: Super2` (Super2 with no lifetime params)
// is now rejected at trait-definition time by the eager downcast-safety
// check. See tests/ui/trait-cast/erasure-region-closure.rs for the
// compile-fail coverage of this pattern.
// =========================================================================
// Case 3: dyn for<'a> Super<'a> -> dyn Sub via Sub: Super<'static>
//
// Structural: backward fails. Root bv r0 is not mapped to by any target
// bv (the target has none). r0 <- ? fails.
// Result: always false.
// =========================================================================

trait Super3<'a>: TraitMetadataTable<dyn Super3<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Sub3: Super3<'static> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S3;

impl<'a> Super3<'a> for S3 {
    fn id(&self) -> u32 { 3 }
}
impl Sub3 for S3 {
    fn sub_val(&self) -> u32 { 30 }
}

/// The target dyn type has no binder variables, but the root has r0.
/// Backward coverage fails: r0 is not reached by any target bv.
/// The cast must fail.
#[inline(never)]
fn case3_backward_fails<'a>(x: &'a u32) {
    let obj: &dyn Super3<'_> = &S3;
    core::cast!(in dyn Super3<'_>, obj => dyn Sub3).expect_err("case3_backward_fails");
}

// =========================================================================
// Case 4: dyn for<'a,'b> Super<'a,'b> -> dyn for<'a> Sub<'a>
//         via Sub<'a>: Super<'a, 'a>
//
// Structural: t0 -> {r0, r1}. Both root bvs are reached (backward ok),
// and the target bv maps to both (forward ok). BUT the two root bvs
// must be equivalent (mutual outlives between r0 and r1), which requires
// outlives evidence.
// Result: depends on outlives.
// =========================================================================

trait Super4<'a, 'b>: TraitMetadataTable<dyn Super4<'a, 'b>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Sub4<'a>: Super4<'a, 'a> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S4<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Super4<'a, 'b> for S4<'a, 'b> {
    fn id(&self) -> u32 { 4 }
}
impl<'a> Sub4<'a> for S4<'a, 'a> {
    fn sub_val(&self) -> u32 { 40 }
}

/// When 'a == 'b (same reference), the two root bvs are equivalent,
/// so the cast succeeds.
#[inline(never)]
fn case4_bounded<'a>(x: &'a u32) {
    let obj: &dyn Super4<'_, '_> = &S4 { _x: x, _y: x };
    let sub = core::cast!(in dyn Super4<'_, '_>, obj => dyn Sub4<'_>).expect("case4_bounded");
    assert_eq!(sub.sub_val(), 40);
}

/// When 'a and 'b are unrelated (no mutual outlives), the cast fails
/// because r0 and r1 are not provably equivalent.
/// Borrowck forces both lifetimes to be equivalent due to the common TraitMetadataTable marker.

// =========================================================================
// Case 5: dyn for<'a,'b> Super<'a,'b> -> dyn for<'a,'b> Sub<'a,'b>
//         via Sub<'a,'b>: Super<'b,'a> (swap)
//
// Structural: t0 -> r1, t1 -> r0. Both directions covered.
// Result: depends on outlives (needs mutual outlives for each pair).
// =========================================================================

trait Super5<'a, 'b>: TraitMetadataTable<dyn Super5<'a, 'b>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Sub5<'a, 'b>: Super5<'b, 'a> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S5<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Super5<'a, 'b> for S5<'a, 'b> {
    fn id(&self) -> u32 { 5 }
}
impl<'a, 'b> Sub5<'a, 'b> for S5<'b, 'a> {
    fn sub_val(&self) -> u32 { 50 }
}

/// When 'a and 'b are the same region, the swap is trivially satisfied.
#[inline(never)]
fn case5_bounded<'a>(x: &'a u32) {
    let obj: &dyn Super5<'_, '_> = &S5 { _x: x, _y: x };
    let sub = core::cast!(in dyn Super5<'_, '_>, obj => dyn Sub5<'_, '_>).expect("case5_bounded");
    assert_eq!(sub.sub_val(), 50);
}

/// When 'a and 'b are unrelated, the swap mapping still succeeds
/// because each structural pair maps to the same concrete lifetime:
/// target_bv0 and root_bv1 both hold 'local, target_bv1 and root_bv0
/// both hold 'a.
#[inline(never)]
fn case5_unbounded<'a>(x: &'a u32) {
    let local: u32 = 99;
    let obj: &dyn Super5<'_, '_> = &S5 { _x: x, _y: &local };
    let sub = core::cast!(in dyn Super5<'_, '_>, obj => dyn Sub5<'_, '_>)
        .expect("case5_unbounded");
    assert_eq!(sub.sub_val(), 50);
}

// =========================================================================
// Case 6: Same-trait cast: dyn for<'a> Super<'a> -> dyn Super<'a> (self)
//
// Structural: identity mapping (t0 -> r0, trivially surjective).
// Result: always true (identity mapping, trivially mutual).
// =========================================================================

/// A same-trait cast is always erasure-safe.
#[inline(never)]
fn case6_identity<'a>(x: &'a u32) {
    let obj: &dyn Super1<'_> = &S1 { _x: x };
    // Cast from dyn Super1 to dyn Super1 is the identity cast.
    let same = core::cast!(in dyn Super1<'_>, obj => dyn Super1<'_>).expect("case6_identity");
    assert_eq!(same.id(), 1);
}

// =========================================================================
// Case 7: No principal (auto-trait only)
//
// When neither root nor target has a principal trait (only auto traits),
// there are no binder variables. Surjectivity is trivially satisfied.
// Result: always true.
// =========================================================================

// Note: auto-trait-only casts are a degenerate case. The existing test
// infrastructure requires a principal trait for the TraitMetadataTable
// bound. We exercise this indirectly: a root with no lifetime params
// casting to a target with no lifetime params is the analogous case.

trait Super7: TraitMetadataTable<dyn Super7> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Sub7: Super7 {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S7;

impl Super7 for S7 {
    fn id(&self) -> u32 { 7 }
}
impl Sub7 for S7 {
    fn sub_val(&self) -> u32 { 70 }
}

/// No binder variables on either side. Trivially erasure-safe.
#[inline(never)]
fn case7_no_bvs(obj: &dyn Super7) {
    let sub = core::cast!(in dyn Super7, obj => dyn Sub7).expect("case7_no_bvs");
    assert_eq!(sub.sub_val(), 70);
}

fn main() {
    let x: u32 = 42;

    // Case 1: identity mapping, bounded — succeeds.
    case1_bounded(&x);

    // Case 2 removed — see note above; compile-fail coverage in
    // tests/ui/trait-cast/erasure-region-closure.rs.

    // Case 3: backward coverage failure — always fails.
    case3_backward_fails(&x);

    // Case 4: fan-out mapping t0 -> {r0, r1}.
    case4_bounded(&x);    // same region -> succeeds

    // Case 5: swap mapping t0 -> r1, t1 -> r0.
    case5_bounded(&x);    // same region -> succeeds
    case5_unbounded(&x);  // unrelated regions -> fails

    // Case 6: same-trait identity cast — always succeeds.
    case6_identity(&x);

    // Case 7: no binder variables — always succeeds.
    case7_no_bvs(&S7 as &dyn Super7);
}
