//@ run-pass
//! Erasure-safety projection predicate tests.
//!
//! Binder variables can appear not only in the principal trait's generic
//! args, but also in associated type projections (e.g., `Assoc = &'a u8`).
//! The erasure-safety check must account for these by matching projections
//! between the target and root dyn types by associated type DefId, then
//! walking both in TypeVisitor DFS order to establish bv correspondence.
//!
//! Test cases:
//! - Projection match: both root and target dyn types carry the same
//!   projection -> bv correspondence through projection.
//! - Principal + projection bvs: binder variables appear in both the
//!   principal trait args and the projection term.
//! - Transitive chain with projection: the projection flows through
//!   an intermediate supertrait.

#![feature(trait_cast)]

#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Case 1: Projection match through associated types.
//
// The root trait Root1 has an associated type Assoc. When used as
// `dyn Root1<Assoc = &'a u8>`, the projection creates a binder variable.
// The sub-trait Sub1 inherits from Root1, so `dyn Sub1<Assoc = &'a u8>`
// carries the same projection. The bv correspondence is established
// through the shared projection predicate.
// =========================================================================

trait Root1: TraitMetadataTable<dyn Root1<Assoc = Self::Assoc>> + core::fmt::Debug {
    type Assoc;
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Sub1: Root1 {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct P1<'a> { _x: &'a u8 }

impl<'a> Root1 for P1<'a> {
    type Assoc = &'a u8;
    fn id(&self) -> u32 { 1 }
}
impl<'a> Sub1 for P1<'a> {
    fn val(&self) -> u32 { 10 }
}

/// Both `dyn Root1<Assoc = &'a u8>` and `dyn Sub1<Assoc = &'a u8>` have
/// a binder variable for 'a in the projection. The structural check maps
/// the projection's bv in the target to the same bv in the root.
#[inline(never)]
fn case1_projection_match<'a>(x: &'a u8) {
    let obj: &dyn Root1<Assoc = &'a u8> = &P1 { _x: x };
    let sub = core::cast!(
        in dyn Root1<Assoc = &'a u8> + '_,
        obj => dyn Sub1<Assoc = &'a u8> + '_
    ).expect("case1_projection_match");
    assert_eq!(sub.val(), 10);
}

// =========================================================================
// Case 2: Principal trait args carry a lifetime + projection carries
// the same lifetime.
//
// Root2<'a> has Assoc = &'a u8. When used as
// `dyn Root2<'a, Assoc = &'a u8>`, the principal arg and the projection
// both reference the same binder variable. The sub-trait Sub2<'a>
// inherits from Root2<'a>, producing the same projection.
// =========================================================================

trait Root2<'a>: TraitMetadataTable<dyn Root2<'a, Assoc = &'a u8>> + core::fmt::Debug {
    type Assoc;
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Sub2<'a>: Root2<'a> {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct P2<'a> { _x: &'a u8 }

impl<'a> Root2<'a> for P2<'a> {
    type Assoc = &'a u8;
    fn id(&self) -> u32 { 2 }
}
impl<'a> Sub2<'a> for P2<'a> {
    fn val(&self) -> u32 { 20 }
}

/// The principal (Root2<'a>) and projection (Assoc = &'a u8) carry
/// lifetimes that refer to the same underlying bv. Cast succeeds.
#[inline(never)]
fn case2_principal_and_projection<'a>(x: &'a u8) {
    let obj: &dyn Root2<'_, Assoc = &'a u8> = &P2 { _x: x };
    let sub = core::cast!(
        in dyn Root2<'_, Assoc = &'a u8>,
        obj => dyn Sub2<'_, Assoc = &'a u8>
    ).expect("case2_principal_and_projection");
    assert_eq!(sub.val(), 20);
}

// =========================================================================
// Case 3: Transitive chain with projection.
//
// Sub3: Mid3: Root3, where Root3 defines an associated type. The
// projection `Assoc = &'a u8` is specified in the dyn type and must
// be matched at each level of the chain.
// =========================================================================

trait Root3: TraitMetadataTable<dyn Root3<Assoc = Self::Assoc>> + core::fmt::Debug {
    type Assoc;
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Mid3: Root3 {
    fn mid_val(&self) -> u32;
}
trait Sub3: Mid3 {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct P3<'a> { _x: &'a u8 }

impl<'a> Root3 for P3<'a> {
    type Assoc = &'a u8;
    fn id(&self) -> u32 { 3 }
}
impl<'a> Mid3 for P3<'a> {
    fn mid_val(&self) -> u32 { 30 }
}
impl<'a> Sub3 for P3<'a> {
    fn sub_val(&self) -> u32 { 31 }
}

/// The projection flows from Root3 through Mid3 to Sub3. Casting
/// through the chain preserves the bv correspondence.
#[inline(never)]
fn case3_transitive_projection<'a>(x: &'a u8) {
    let obj: &dyn Root3<Assoc = &'a u8> = &P3 { _x: x };

    // Cast to Mid3 (depth-1).
    let mid = core::cast!(
        in dyn Root3<Assoc = &'a u8>,
        obj => dyn Mid3<Assoc = &'a u8>
    ).expect("case3_transitive_projection: mid");
    assert_eq!(mid.mid_val(), 30);

    // Cast to Sub3 (depth-2).
    let sub = core::cast!(
        in dyn Root3<Assoc = &'a u8>,
        obj => dyn Sub3<Assoc = &'a u8>
    ).expect("case3_transitive_projection: sub");
    assert_eq!(sub.sub_val(), 31);
}

// =========================================================================
// Case 4: No-lifetime traits with projections (no bvs).
//
// When the associated type doesn't involve any lifetimes (e.g.,
// `Assoc = u32`), the projections carry no binder variables.
// Surjectivity is trivially satisfied.
// =========================================================================

trait Root4: TraitMetadataTable<dyn Root4<Assoc = u32>> + core::fmt::Debug {
    type Assoc;
    #[allow(dead_code)]
    fn id(&self) -> u32;
}
trait Sub4: Root4 {
    fn val(&self) -> u32;
}

#[derive(Debug)]
struct P4;

impl Root4 for P4 {
    type Assoc = u32;
    fn id(&self) -> u32 { 4 }
}
impl Sub4 for P4 {
    fn val(&self) -> u32 { 40 }
}

/// No binder variables in the projection. Trivially erasure-safe.
#[inline(never)]
fn case4_no_lifetime_projection(obj: &dyn Root4<Assoc = u32>) {
    let sub = core::cast!(
        in dyn Root4<Assoc = u32>,
        obj => dyn Sub4<Assoc = u32>
    ).expect("case4_no_lifetime_projection");
    assert_eq!(sub.val(), 40);
}

fn main() {
    let x: u8 = 42;

    // Case 1: projection match, same bv.
    case1_projection_match(&x);

    // Case 2: principal + projection bvs.
    case2_principal_and_projection(&x);

    // Case 3: transitive projection chain.
    case3_transitive_projection(&x);

    // Case 4: no-lifetime projection (trivially safe).
    case4_no_lifetime_projection(&P4 as &dyn Root4<Assoc = u32>);
}
