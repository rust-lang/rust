//@ run-pass
//! Erasure-safety supertrait chain walk tests (S14.3.4).
//!
//! The structural surjectivity check must walk the supertrait chain from
//! the target trait up to the root trait, mapping binder variables at each
//! step. These tests exercise:
//!
//! - Depth-1: Sub directly extends Super.
//! - Depth-2: Sub extends Mid extends Super (transitive mapping).
//! - Diamond: Sub extends Mid1 + Mid2, both extend Super. The same root
//!   bv must be reachable through both paths consistently.
//!
//! All tests use lifetime-parameterized traits to exercise bv mapping
//! through the chain.

#![feature(trait_cast)]
#![allow(dead_code, unused_variables)]

#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Case 1: Depth-1 chain. Sub<'a>: Super<'a>
//
// The simplest chain walk: one step from target to root.
// bv mapping: t0 -> r0 directly.
// =========================================================================

trait Super1<'a>: TraitMetadataTable<dyn Super1<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Sub1<'a>: Super1<'a> {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct C1<'a> { _x: &'a u32 }

impl<'a> Super1<'a> for C1<'a> {
    fn id(&self) -> u32 { 1 }
}
impl<'a> Sub1<'a> for C1<'a> {
    fn val(&self) -> u32 { 10 }
}

#[inline(never)]
fn case1_depth1<'a>(x: &'a u32) {
    let obj: &dyn Super1<'_> = &C1 { _x: x };
    let sub = core::cast!(in dyn Super1<'_>, obj => dyn Sub1<'_>).expect("case1_depth1");
    assert_eq!(sub.val(), 10);
}

// =========================================================================
// Case 2: Depth-2 chain. Sub<'a>: Mid<'a>: Super<'a>
//
// Two-step chain walk: target -> mid -> root.
// bv mapping: t0 -> mid0 -> r0 (transitive).
// =========================================================================

trait Super2<'a>: TraitMetadataTable<dyn Super2<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Mid2<'a>: Super2<'a> {
    fn mid_val(&self) -> u32;
}
trait Sub2<'a>: Mid2<'a> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C2<'a> { _x: &'a u32 }

impl<'a> Super2<'a> for C2<'a> {
    fn id(&self) -> u32 { 2 }
}
impl<'a> Mid2<'a> for C2<'a> {
    fn mid_val(&self) -> u32 { 20 }
}
impl<'a> Sub2<'a> for C2<'a> {
    fn sub_val(&self) -> u32 { 21 }
}

#[inline(never)]
fn case2_depth2<'a>(x: &'a u32) {
    let obj: &dyn Super2<'_> = &C2 { _x: x };

    // Direct cast to Mid2 (depth-1).
    let mid = core::cast!(in dyn Super2<'_>, obj => dyn Mid2<'_>).expect("case2_depth2: mid");
    assert_eq!(mid.mid_val(), 20);

    // Cast to Sub2 (depth-2, transitive chain walk).
    let sub = core::cast!(in dyn Super2<'_>, obj => dyn Sub2<'_>).expect("case2_depth2: sub");
    assert_eq!(sub.sub_val(), 21);
}

// =========================================================================
// Case 3: Depth-2 with lifetime transformation.
//
// Sub<'a,'b>: Mid<'a,'b>: Super<'a,'b>
// The chain walk passes two lifetime params through two steps.
// =========================================================================

trait Super3<'a, 'b>: TraitMetadataTable<dyn Super3<'a, 'b>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Mid3<'a, 'b>: Super3<'a, 'b> {
    fn mid_val(&self) -> u32;
}
trait Sub3<'a, 'b>: Mid3<'a, 'b> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C3<'a, 'b> { _x: &'a u32, _y: &'b u32 }

impl<'a, 'b> Super3<'a, 'b> for C3<'a, 'b> {
    fn id(&self) -> u32 { 3 }
}
impl<'a, 'b> Mid3<'a, 'b> for C3<'a, 'b> {
    fn mid_val(&self) -> u32 { 30 }
}
impl<'a, 'b> Sub3<'a, 'b> for C3<'a, 'b> {
    fn sub_val(&self) -> u32 { 31 }
}

#[inline(never)]
fn case3_depth2_multi_lifetime<'a>(x: &'a u32) {
    // Both lifetimes are the same -> cast succeeds.
    let obj: &dyn Super3<'_, '_> = &C3 { _x: x, _y: x };
    let sub = core::cast!(in dyn Super3<'_, '_>, obj => dyn Sub3<'_, '_>)
        .expect("case3_depth2_multi_lifetime");
    assert_eq!(sub.sub_val(), 31);
}

#[inline(never)]
fn case3_depth2_multi_lifetime_mixed<'a>(x: &'a u32) {
    // Different lifetimes -> cast to Sub3 with both still works (they're
    // separate bvs, the mapping is t0->r0, t1->r1 with no merging).
    let local: u32 = 99;
    let obj: &dyn Super3<'_, '_> = &C3 { _x: x, _y: &local };
    let sub = core::cast!(in dyn Super3<'_, '_>, obj => dyn Sub3<'_, '_>)
        .expect("case3_depth2_multi_lifetime_mixed");
    assert_eq!(sub.sub_val(), 31);
}

// =========================================================================
// Case 4: Diamond inheritance.
//
// Sub: Mid1<'a> + Mid2<'a>, where Mid1<'a>: Super<'a> and Mid2<'a>: Super<'a>
//
// The root bv r0 is reachable through both Mid1 and Mid2. The chain
// walk must find the same root bv through both paths.
// =========================================================================

trait Super4<'a>: TraitMetadataTable<dyn Super4<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Mid4A<'a>: Super4<'a> {
    fn mid_a_val(&self) -> u32;
}
trait Mid4B<'a>: Super4<'a> {
    fn mid_b_val(&self) -> u32;
}
trait Sub4<'a>: Mid4A<'a> + Mid4B<'a> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C4<'a> { _x: &'a u32 }

impl<'a> Super4<'a> for C4<'a> {
    fn id(&self) -> u32 { 4 }
}
impl<'a> Mid4A<'a> for C4<'a> {
    fn mid_a_val(&self) -> u32 { 40 }
}
impl<'a> Mid4B<'a> for C4<'a> {
    fn mid_b_val(&self) -> u32 { 41 }
}
impl<'a> Sub4<'a> for C4<'a> {
    fn sub_val(&self) -> u32 { 42 }
}

#[inline(never)]
fn case4_diamond<'a>(x: &'a u32) {
    let obj: &dyn Super4<'_> = &C4 { _x: x };

    // Cast through Mid4A path.
    let mid_a = core::cast!(in dyn Super4<'_>, obj => dyn Mid4A<'_>).expect("case4_diamond: mid_a");
    assert_eq!(mid_a.mid_a_val(), 40);

    // Cast through Mid4B path.
    let mid_b = core::cast!(in dyn Super4<'_>, obj => dyn Mid4B<'_>).expect("case4_diamond: mid_b");
    assert_eq!(mid_b.mid_b_val(), 41);

    // Cast to Sub4 (diamond join: both Mid4A and Mid4B paths reach Super4).
    let sub = core::cast!(in dyn Super4<'_>, obj => dyn Sub4<'_>).expect("case4_diamond: sub");
    assert_eq!(sub.sub_val(), 42);
}

// =========================================================================
// Case 5: Diamond with lifetime-bounded impl on one branch.
//
// Sub: Mid4A<'a> + Mid4B<'a>, but Sub's impl requires 'a: 'static on
// one of the mid-trait paths. This tests that the diamond walk
// correctly handles cases where one path is more constrained.
// =========================================================================

trait Super5<'a>: TraitMetadataTable<dyn Super5<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Mid5A<'a>: Super5<'a> {
    fn mid_a_val(&self) -> u32;
}
trait Mid5B<'a>: Super5<'a> {
    fn mid_b_val(&self) -> u32;
}

#[derive(Debug)]
struct C5<'a> { _x: &'a u32 }

impl<'a> Super5<'a> for C5<'a> {
    fn id(&self) -> u32 { 5 }
}
impl<'a> Mid5A<'a> for C5<'a> {
    fn mid_a_val(&self) -> u32 { 50 }
}
// Mid5B only implemented with a where clause
impl<'a> Mid5B<'a> for C5<'a>
where
    'a: 'static,
{
    fn mid_b_val(&self) -> u32 { 51 }
}

/// 'a is a local non-static lifetime. Mid5A is always available, but
/// Mid5B requires 'a: 'static which doesn't hold.
#[inline(never)]
fn case5_diamond_constrained<'a>(x: &'a u32) {
    let obj: &dyn Super5<'_> = &C5 { _x: x };

    // Mid5A cast succeeds (no extra constraints).
    let mid_a = core::cast!(in dyn Super5<'_>, obj => dyn Mid5A<'_>)
        .expect("case5_diamond_constrained: mid_a");
    assert_eq!(mid_a.mid_a_val(), 50);

    // Mid5B cast fails (requires 'a: 'static, which doesn't hold).
    core::cast!(in dyn Super5<'_>, obj => dyn Mid5B<'_>)
        .expect_err("case5_diamond_constrained: mid_b");
}

// =========================================================================
// Case 6: Depth-3 chain. Sub: Mid2: Mid1: Super
//
// Three-step chain walk for completeness.
// =========================================================================

trait Super6<'a>: TraitMetadataTable<dyn Super6<'a>> + core::fmt::Debug {
    fn id(&self) -> u32;
}
trait Mid6A<'a>: Super6<'a> {
    fn a_val(&self) -> u32;
}
trait Mid6B<'a>: Mid6A<'a> {
    fn b_val(&self) -> u32;
}
trait Sub6<'a>: Mid6B<'a> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C6<'a> { _x: &'a u32 }

impl<'a> Super6<'a> for C6<'a> {
    fn id(&self) -> u32 { 6 }
}
impl<'a> Mid6A<'a> for C6<'a> {
    fn a_val(&self) -> u32 { 60 }
}
impl<'a> Mid6B<'a> for C6<'a> {
    fn b_val(&self) -> u32 { 61 }
}
impl<'a> Sub6<'a> for C6<'a> {
    fn sub_val(&self) -> u32 { 62 }
}

#[inline(never)]
fn case6_depth3<'a>(x: &'a u32) {
    let obj: &dyn Super6<'_> = &C6 { _x: x };

    // Cast through each depth level.
    let mid_a = core::cast!(in dyn Super6<'_>, obj => dyn Mid6A<'_>).expect("case6_depth3: mid_a");
    assert_eq!(mid_a.a_val(), 60);

    let mid_b = core::cast!(in dyn Super6<'_>, obj => dyn Mid6B<'_>).expect("case6_depth3: mid_b");
    assert_eq!(mid_b.b_val(), 61);

    let sub = core::cast!(in dyn Super6<'_>, obj => dyn Sub6<'_>).expect("case6_depth3: sub");
    assert_eq!(sub.sub_val(), 62);
}

fn main() {
    let x: u32 = 42;

    // Case 1: depth-1 direct mapping.
    case1_depth1(&x);

    // Case 2: depth-2 transitive mapping.
    case2_depth2(&x);

    // Case 3: depth-2 with multiple lifetimes.
    case3_depth2_multi_lifetime(&x);
    case3_depth2_multi_lifetime_mixed(&x);

    // Case 4: diamond inheritance.
    case4_diamond(&x);

    // Case 5: diamond with constrained branch.
    case5_diamond_constrained(&x);

    // Case 6: depth-3 chain.
    case6_depth3(&x);
}
