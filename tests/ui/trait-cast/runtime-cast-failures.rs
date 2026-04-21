//@ run-pass
//! Runtime trait-cast failures: patterns from lifetime-in-generics.rs
//! where `cast!` is expected to return `Err` at runtime.
//!
//! Existing tests already cover single missing impls (basic-downcast,
//! torture-tests case 1 sparse), single where-clause gating with
//! direct lifetime params (lifetime-bounded-downcast), diamond with a
//! where-clause-gated branch (erasure-safety-chain-walk case 5), and
//! chain with where-clause-gated impls (torture-tests case 5).
//!
//! This file focuses on negative patterns that are NOT yet covered:
//!
//!   1. Transitive where-clause chain `'a: 'b, 'b: 'c` where the
//!      middle link is unprovable.  Unlike the single-bound gating in
//!      lifetime-bounded-downcast, this tests multi-bound composition.
//!   2. Chain where the intermediate sub-trait impl is literally
//!      MISSING (not where-clause gated).  Complements torture-tests
//!      case 5 which tests where-clause gating in the same shape.
//!   3. Projection + where-clause gating combined: the sub-trait impl
//!      is gated by a lifetime bound when the trait also carries an
//!      associated-type projection.  Complements erasure-safety-
//!      projections which has no negative cases.
//!   4. Multi-type-param diamond with one branch impl missing:
//!      `Root<T, U>` → `MidT<T, U>` / `MidU<T, U>` → `Sub<T, U>`,
//!      where only MidT is impled.  No existing test uses a two-type-
//!      param diamond for negative cases.
//!
//! All cases use the type-param-with-lifetime pattern from
//! lifetime-in-generics.rs (e.g., `Root<&'a u32>`), which the existing
//! negative tests do not exercise.

#![feature(trait_cast)]
#![allow(dead_code, unused_variables)]

#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Case 1: Transitive where-clause chain, middle link unprovable.
//
// The impl requires `'a: 'b` AND `'b: 'c`.  The coercion site has
// 'b as a strictly interior scope, so neither bound is provable.
// Existing where-clause tests (lifetime-bounded-downcast, torture
// case 5) use a single bound on direct lifetime params; this case
// tests multi-bound composition with lifetimes embedded in type params.
// =========================================================================

trait Root1<T, U, V>: TraitMetadataTable<dyn Root1<T, U, V>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait Sub1<T, U, V>: Root1<T, U, V> {
    fn chained(&self) -> u32;
}

#[derive(Debug)]
struct S1<'a, 'b, 'c> { a: &'a u32, b: &'b u32, c: &'c u32 }

impl<'a, 'b, 'c> Root1<&'a u32, &'b u32, &'c u32> for S1<'a, 'b, 'c> {
    fn val(&self) -> u32 { *self.a + *self.b + *self.c }
}
impl<'a, 'b, 'c> Sub1<&'a u32, &'b u32, &'c u32> for S1<'a, 'b, 'c>
where
    'a: 'b,
    'b: 'c,
{
    fn chained(&self) -> u32 { *self.a * *self.b * *self.c }
}

/// 'b is strictly interior — neither `'a: 'b` nor `'b: 'c` is provable.
#[inline(never)]
fn case1_chain_unprovable<'a>(x: &'a u32) {
    let b_scope: u32 = 2;
    let c_scope: u32 = 3;
    let obj: &dyn Root1<&'_ u32, &'_ u32, &'_ u32> =
        &S1 { a: x, b: &b_scope, c: &c_scope };
    core::cast!(
        in dyn Root1<&'_ u32, &'_ u32, &'_ u32>,
        obj => dyn Sub1<&'_ u32, &'_ u32, &'_ u32>
    ).expect_err("case1_chain_unprovable");
}

// =========================================================================
// Case 2: Chain with intermediate impl literally missing.
//
// Root2<T> → Mid2<T> → Sub2<T>, with T = &'a u32.  Only Root2 is
// implemented for the concrete type — Mid2 and Sub2 have no impls at
// all.  This is a different failure mechanism from torture-tests
// case 5 (where Mid and Sub impls exist but are where-clause gated).
// =========================================================================

trait Root2<T>: TraitMetadataTable<dyn Root2<T>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait Mid2<T>: Root2<T> {
    fn mid_val(&self) -> u32;
}
trait Sub2<T>: Mid2<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S2<'a> { x: &'a u32 }

impl<'a> Root2<&'a u32> for S2<'a> {
    fn val(&self) -> u32 { *self.x }
}
// NOTE: no Mid2 or Sub2 impl.

#[inline(never)]
fn case2_chain_missing<'a>(x: &'a u32) {
    let obj: &dyn Root2<&'_ u32> = &S2 { x };
    core::cast!(in dyn Root2<&'_ u32>, obj => dyn Mid2<&'_ u32>)
        .expect_err("case2_mid_missing");
    core::cast!(in dyn Root2<&'_ u32>, obj => dyn Sub2<&'_ u32>)
        .expect_err("case2_sub_missing");
}

// =========================================================================
// Case 3: Projection + where-clause gating.
//
// Root3<T, Assoc = T> has an associated-type projection.  Sub3's impl
// is gated by a lifetime bound `'b: 'a` on a lifetime that is not
// visible in the projection.  erasure-safety-projections has no
// negative cases — this case shows a projection-carrying dyn type
// whose sub-trait cast can still fail at runtime.
// =========================================================================

trait Root3<T>: TraitMetadataTable<dyn Root3<T, Assoc = T>> + core::fmt::Debug {
    type Assoc;
    fn val(&self) -> u32;
}
trait Sub3<T>: Root3<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S3<'a, 'b> { x: &'a u32, y: &'b u32 }

impl<'a, 'b> Root3<&'a u32> for S3<'a, 'b> {
    type Assoc = &'a u32;
    fn val(&self) -> u32 { *self.x }
}
impl<'a, 'b> Sub3<&'a u32> for S3<'a, 'b>
where
    'b: 'a,
{
    fn sub_val(&self) -> u32 { *self.y }
}

#[inline(never)]
fn case3_projection_where<'a>(x: &'a u32) {
    let local: u32 = 5;
    let obj: &dyn Root3<&'a u32, Assoc = &'a u32> = &S3 { x, y: &local };
    core::cast!(
        in dyn Root3<&'a u32, Assoc = &'a u32>,
        obj => dyn Sub3<&'a u32, Assoc = &'a u32>
    ).expect_err("case3_projection_where");
}

// =========================================================================
// Case 4: Multi-type-param diamond with one branch impl missing.
//
// Mirrors lifetime-in-generics case 11 (Root<T, U>, MidT, MidU, Sub).
// Only MidT is implemented; MidU and the diamond Sub are not.  No
// existing negative test uses a two-type-param diamond.
// =========================================================================

trait Root4<T, U>: TraitMetadataTable<dyn Root4<T, U>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait MidT4<T, U>: Root4<T, U> {
    fn focus_t(&self) -> u32;
}
trait MidU4<T, U>: Root4<T, U> {
    fn focus_u(&self) -> u32;
}
trait Sub4<T, U>: MidT4<T, U> + MidU4<T, U> {
    fn combined(&self) -> u32;
}

#[derive(Debug)]
struct S4<'a, 'b> { t: &'a u32, u: &'b u32 }

impl<'a, 'b> Root4<&'a u32, &'b u32> for S4<'a, 'b> {
    fn val(&self) -> u32 { *self.t + *self.u }
}
impl<'a, 'b> MidT4<&'a u32, &'b u32> for S4<'a, 'b> {
    fn focus_t(&self) -> u32 { *self.t }
}
// NOTE: no MidU4 impl, no Sub4 impl.

#[inline(never)]
fn case4_diamond_missing<'a>(x: &'a u32) {
    let obj: &dyn Root4<&'_ u32, &'_ u32> = &S4 { t: x, u: x };

    // Positive: MidT4 is impled.
    let mt = core::cast!(
        in dyn Root4<&'_ u32, &'_ u32>,
        obj => dyn MidT4<&'_ u32, &'_ u32>
    ).expect("case4_mt");
    assert_eq!(mt.focus_t(), *x);

    // Negative: MidU4 has no impl.
    core::cast!(
        in dyn Root4<&'_ u32, &'_ u32>,
        obj => dyn MidU4<&'_ u32, &'_ u32>
    ).expect_err("case4_mu_missing");

    // Negative: Sub4 requires both branches.
    core::cast!(
        in dyn Root4<&'_ u32, &'_ u32>,
        obj => dyn Sub4<&'_ u32, &'_ u32>
    ).expect_err("case4_sub_missing");
}

fn main() {
    let x: u32 = 10;

    eprintln!("=== case1: transitive where-clause unprovable ===");
    case1_chain_unprovable(&x);

    eprintln!("=== case2: chain links missing ===");
    case2_chain_missing(&x);

    eprintln!("=== case3: projection + where-clause ===");
    case3_projection_where(&x);

    eprintln!("=== case4: multi-type-param diamond missing ===");
    case4_diamond_missing(&x);
}
