//@ run-pass
//! Trait-cast with lifetimes embedded inside generic *type* parameters.
//!
//! The existing tests use lifetime parameters on traits (e.g., Root<'a>).
//! These tests use generic *type* parameters whose instantiations carry
//! lifetimes (e.g., Root<&'a u32>, Root<Option<&'a u32>>).  The dyn
//! types carry their lifetimes inside type arguments rather than as
//! binder variables.
//!
//! Test cases:
//!   1. Single type param: Root<T> with T = &'a u32.
//!   2. Two type params merged: Root<T, U> → Sub<T>: Root<T, T>.
//!   3. Where-clause gating: Root<T, U> with hidden lifetime constraint.
//!   4. Nested generic: Root<T> with T = Option<&'a u32>.
//!   5. Depth-3 chain with type param: Sub<T>: Mid<T>: Root<T>.
//!   6. Diamond with type param: Sub<T>: Mid1<T> + Mid2<T>, both: Root<T>.
//!   7. Mixed lifetime + type params: Root<'a, T>.
//!   8. Type param + projected associated type: Root<T, Assoc = T>.
//!   9. Type param + projected transformation: Root<T, Wrapped = Option<T>>.
//!  10. Chain + diamond combined: 4-level hierarchy.
//!  11. Multi-type-param diamond: Sub<T,U>: MidT<T,U> + MidU<T,U>.
//!
//! Note: GATs with sub-trait casting ICE in well-formedness checking,
//! which appears to be a compiler bug independent of the trait-cast
//! machinery; the `Container::Lent<'a>` shape is therefore not
//! exercised here.

#![feature(trait_cast)]
#![allow(dead_code, unused_variables)]

#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Case 1: trait Root<T> with T = &'a u32
//
// The lifetime 'a is embedded inside the type parameter, not a direct
// lifetime param on the trait.  The dyn type is dyn Root1<&'a u32>.
// =========================================================================

trait Root1<T>: TraitMetadataTable<dyn Root1<T>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait Sub1<T>: Root1<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S1<'a> { x: &'a u32 }

impl<'a> Root1<&'a u32> for S1<'a> {
    fn val(&self) -> u32 { *self.x }
}
impl<'a> Sub1<&'a u32> for S1<'a> {
    fn sub_val(&self) -> u32 { *self.x + 1 }
}

#[inline(never)]
fn case1<'a>(x: &'a u32) {
    let obj: &dyn Root1<&'_ u32> = &S1 { x };
    let sub = core::cast!(in dyn Root1<&'_ u32>, obj => dyn Sub1<&'_ u32>)
        .expect("case1");
    assert_eq!(sub.sub_val(), *x + 1);
}

// =========================================================================
// Case 2: Two type params — Sub merges both
//
// Root2<T, U> with T = &'a u32, U = &'b u32.
// Sub2<T>: Root2<T, T> collapses both params into one.
// =========================================================================

trait Root2<T, U>: TraitMetadataTable<dyn Root2<T, U>> + core::fmt::Debug {
    fn first(&self) -> u32;
    fn second(&self) -> u32;
}
trait Sub2<T>: Root2<T, T> {
    fn merged(&self) -> u32;
}

#[derive(Debug)]
struct S2<'a, 'b> { x: &'a u32, y: &'b u32 }

impl<'a, 'b> Root2<&'a u32, &'b u32> for S2<'a, 'b> {
    fn first(&self) -> u32 { *self.x }
    fn second(&self) -> u32 { *self.y }
}
impl<'a> Sub2<&'a u32> for S2<'a, 'a> {
    fn merged(&self) -> u32 { *self.x + *self.y }
}

#[inline(never)]
fn case2<'a>(x: &'a u32) {
    let obj: &dyn Root2<&'_ u32, &'_ u32> = &S2 { x, y: x };
    let sub = core::cast!(
        in dyn Root2<&'_ u32, &'_ u32>,
        obj => dyn Sub2<&'_ u32>
    ).expect("case2");
    assert_eq!(sub.merged(), *x + *x);
}

// =========================================================================
// Case 3: Where-clause gating with two type params
//
// Root3<T, U> with T = &'a u32, U = &'b u32.  Sub3Always has no
// where-clause.  Sub3Gated requires 'b: 'a.
// =========================================================================

trait Root3<T, U>: TraitMetadataTable<dyn Root3<T, U>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait Sub3Always<T, U>: Root3<T, U> {
    fn always(&self) -> u32;
}
trait Sub3Gated<T, U>: Root3<T, U> {
    fn gated(&self) -> u32;
}

#[derive(Debug)]
struct S3<'a, 'b> { x: &'a u32, y: &'b u32 }

impl<'a, 'b> Root3<&'a u32, &'b u32> for S3<'a, 'b> {
    fn val(&self) -> u32 { *self.x + *self.y }
}
impl<'a, 'b> Sub3Always<&'a u32, &'b u32> for S3<'a, 'b> {
    fn always(&self) -> u32 { *self.x }
}
impl<'a, 'b> Sub3Gated<&'a u32, &'b u32> for S3<'a, 'b>
where
    'b: 'a,
{
    fn gated(&self) -> u32 { *self.y }
}

/// 'b is a local lifetime, shorter than 'a.
/// Sub3Always succeeds; Sub3Gated fails ('b: 'a not provable).
#[inline(never)]
fn case3<'a>(x: &'a u32) {
    let local: u32 = 5;
    let obj: &dyn Root3<&'_ u32, &'_ u32> = &S3 { x, y: &local };
    let sub = core::cast!(
        in dyn Root3<&'_ u32, &'_ u32>,
        obj => dyn Sub3Always<&'_ u32, &'_ u32>
    ).expect("case3_always");
    assert_eq!(sub.always(), *x);
    core::cast!(
        in dyn Root3<&'_ u32, &'_ u32>,
        obj => dyn Sub3Gated<&'_ u32, &'_ u32>
    ).expect_err("case3_gated");
}

// =========================================================================
// Case 4: Nested generic — T = Option<&'a u32>
//
// The lifetime 'a is nested two levels deep: inside Option, inside
// the trait's type param.  Tests that deeply nested lifetimes are
// tracked through the type parameter.
// =========================================================================

trait Root4<T>: TraitMetadataTable<dyn Root4<T>> + core::fmt::Debug {
    fn inner_val(&self) -> u32;
}
trait Sub4<T>: Root4<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct S4<'a> { opt: Option<&'a u32> }

impl<'a> Root4<Option<&'a u32>> for S4<'a> {
    fn inner_val(&self) -> u32 {
        self.opt.copied().unwrap_or(0)
    }
}
impl<'a> Sub4<Option<&'a u32>> for S4<'a> {
    fn sub_val(&self) -> u32 {
        self.opt.copied().unwrap_or(0) + 10
    }
}

#[inline(never)]
fn case4_some<'a>(x: &'a u32) {
    let obj: &dyn Root4<Option<&'_ u32>> = &S4 { opt: Some(x) };
    let sub = core::cast!(
        in dyn Root4<Option<&'_ u32>>,
        obj => dyn Sub4<Option<&'_ u32>>
    ).expect("case4_some");
    assert_eq!(sub.sub_val(), *x + 10);
}

#[inline(never)]
fn case4_none<'a>(_anchor: &'a u32) {
    let s: S4<'a> = S4 { opt: None };
    let obj: &dyn Root4<Option<&'_ u32>> = &s;
    let sub = core::cast!(
        in dyn Root4<Option<&'_ u32>>,
        obj => dyn Sub4<Option<&'_ u32>>
    ).expect("case4_none");
    assert_eq!(sub.sub_val(), 10);
}

// =========================================================================
// Case 5: Depth-3 chain with type param
//
// Sub5<T>: Mid5<T>: Root5<T>, all with T = &'a u32.  Tests that a
// transitive supertrait chain works when the lifetime is embedded in
// the trait's type parameter.
// =========================================================================

trait Root5<T>: TraitMetadataTable<dyn Root5<T>> + core::fmt::Debug {
    fn root_val(&self) -> u32;
}
trait Mid5<T>: Root5<T> {
    fn mid_val(&self) -> u32;
}
trait Sub5<T>: Mid5<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C5<'a> { x: &'a u32 }

impl<'a> Root5<&'a u32> for C5<'a> {
    fn root_val(&self) -> u32 { *self.x }
}
impl<'a> Mid5<&'a u32> for C5<'a> {
    fn mid_val(&self) -> u32 { *self.x + 100 }
}
impl<'a> Sub5<&'a u32> for C5<'a> {
    fn sub_val(&self) -> u32 { *self.x + 200 }
}

#[inline(never)]
fn case5<'a>(x: &'a u32) {
    let obj: &dyn Root5<&'_ u32> = &C5 { x };
    let mid = core::cast!(in dyn Root5<&'_ u32>, obj => dyn Mid5<&'_ u32>)
        .expect("case5_mid");
    assert_eq!(mid.mid_val(), *x + 100);
    let sub = core::cast!(in dyn Root5<&'_ u32>, obj => dyn Sub5<&'_ u32>)
        .expect("case5_sub");
    assert_eq!(sub.sub_val(), *x + 200);
}

// =========================================================================
// Case 6: Diamond with type param
//
// Sub6<T>: Mid6A<T> + Mid6B<T>, both extending Root6<T>.
// Tests that the diamond join works with type-param lifetimes.
// =========================================================================

trait Root6<T>: TraitMetadataTable<dyn Root6<T>> + core::fmt::Debug {
    fn root_val(&self) -> u32;
}
trait Mid6A<T>: Root6<T> {
    fn a_val(&self) -> u32;
}
trait Mid6B<T>: Root6<T> {
    fn b_val(&self) -> u32;
}
trait Sub6<T>: Mid6A<T> + Mid6B<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C6<'a> { x: &'a u32 }

impl<'a> Root6<&'a u32> for C6<'a> {
    fn root_val(&self) -> u32 { *self.x }
}
impl<'a> Mid6A<&'a u32> for C6<'a> {
    fn a_val(&self) -> u32 { *self.x + 10 }
}
impl<'a> Mid6B<&'a u32> for C6<'a> {
    fn b_val(&self) -> u32 { *self.x + 20 }
}
impl<'a> Sub6<&'a u32> for C6<'a> {
    fn sub_val(&self) -> u32 { *self.x + 30 }
}

#[inline(never)]
fn case6<'a>(x: &'a u32) {
    let obj: &dyn Root6<&'_ u32> = &C6 { x };
    let a = core::cast!(in dyn Root6<&'_ u32>, obj => dyn Mid6A<&'_ u32>)
        .expect("case6_a");
    assert_eq!(a.a_val(), *x + 10);
    let b = core::cast!(in dyn Root6<&'_ u32>, obj => dyn Mid6B<&'_ u32>)
        .expect("case6_b");
    assert_eq!(b.b_val(), *x + 20);
    let sub = core::cast!(in dyn Root6<&'_ u32>, obj => dyn Sub6<&'_ u32>)
        .expect("case6_sub");
    assert_eq!(sub.sub_val(), *x + 30);
}

// =========================================================================
// Case 7: Mixed lifetime param + type param
//
// Root7<'a, T> has both a direct lifetime parameter AND a type
// parameter. The type parameter T = &'b u32 carries another lifetime.
// Tests interaction between binder-variable lifetimes (from 'a) and
// type-embedded lifetimes (from T).
// =========================================================================

trait Root7<'a, T>: TraitMetadataTable<dyn Root7<'a, T>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait Sub7<'a, T>: Root7<'a, T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C7<'a, 'b> { x: &'a u32, y: &'b u32 }

impl<'a, 'b> Root7<'a, &'b u32> for C7<'a, 'b> {
    fn val(&self) -> u32 { *self.x + *self.y }
}
impl<'a, 'b> Sub7<'a, &'b u32> for C7<'a, 'b> {
    fn sub_val(&self) -> u32 { *self.x * *self.y }
}

#[inline(never)]
fn case7<'a>(x: &'a u32) {
    let obj: &dyn Root7<'_, &'_ u32> = &C7 { x, y: x };
    let sub = core::cast!(in dyn Root7<'_, &'_ u32>, obj => dyn Sub7<'_, &'_ u32>)
        .expect("case7");
    assert_eq!(sub.sub_val(), *x * *x);
}

// =========================================================================
// Case 8: Type param + projected associated type
//
// Root8<T> has an associated type Assoc that is constrained to equal T
// in the dyn projection.  Combines the type-param-with-lifetime pattern
// with associated-type projection (similar to erasure-safety-projections).
// =========================================================================

trait Root8<T>: TraitMetadataTable<dyn Root8<T, Assoc = T>> + core::fmt::Debug {
    type Assoc;
    fn val(&self) -> u32;
}
trait Sub8<T>: Root8<T> {
    fn sub_val(&self) -> u32;
}

#[derive(Debug)]
struct C8<'a> { x: &'a u32 }

impl<'a> Root8<&'a u32> for C8<'a> {
    type Assoc = &'a u32;
    fn val(&self) -> u32 { *self.x }
}
impl<'a> Sub8<&'a u32> for C8<'a> {
    fn sub_val(&self) -> u32 { *self.x + 50 }
}

#[inline(never)]
fn case8<'a>(x: &'a u32) {
    let obj: &dyn Root8<&'a u32, Assoc = &'a u32> = &C8 { x };
    let sub = core::cast!(
        in dyn Root8<&'a u32, Assoc = &'a u32>,
        obj => dyn Sub8<&'a u32, Assoc = &'a u32>
    ).expect("case8");
    assert_eq!(sub.sub_val(), *x + 50);
}

// =========================================================================
// Case 9: Type param threaded through both trait params and projections
//
// Root9<T> has an associated type Assoc that contains a transformation
// of T (e.g., Option<T>).  The dyn type's projection fixes the
// transformation.  Sub-trait extends with another transformation.
// Combines: type-param-with-lifetime + projection-with-transformation.
// =========================================================================

trait Root9<T>: TraitMetadataTable<dyn Root9<T, Wrapped = Option<T>>> + core::fmt::Debug {
    type Wrapped;
    fn first_val(&self) -> u32;
}
trait Sub9A<T>: Root9<T> {
    fn a_val(&self) -> u32;
}
trait Sub9B<T>: Root9<T> {
    fn b_val(&self) -> u32;
}

#[derive(Debug)]
struct C9<'a> { x: &'a u32 }

impl<'a> Root9<&'a u32> for C9<'a> {
    type Wrapped = Option<&'a u32>;
    fn first_val(&self) -> u32 { *self.x }
}
impl<'a> Sub9A<&'a u32> for C9<'a> {
    fn a_val(&self) -> u32 { *self.x + 7 }
}
impl<'a> Sub9B<&'a u32> for C9<'a> {
    fn b_val(&self) -> u32 { *self.x + 13 }
}

#[inline(never)]
fn case9<'a>(x: &'a u32) {
    let obj: &dyn Root9<&'a u32, Wrapped = Option<&'a u32>> = &C9 { x };
    let a = core::cast!(
        in dyn Root9<&'a u32, Wrapped = Option<&'a u32>>,
        obj => dyn Sub9A<&'a u32, Wrapped = Option<&'a u32>>
    ).expect("case9_a");
    assert_eq!(a.a_val(), *x + 7);
    let b = core::cast!(
        in dyn Root9<&'a u32, Wrapped = Option<&'a u32>>,
        obj => dyn Sub9B<&'a u32, Wrapped = Option<&'a u32>>
    ).expect("case9_b");
    assert_eq!(b.b_val(), *x + 13);
}

// =========================================================================
// Case 10: Chain + diamond combined — 4-level hierarchy
//
// Root10<T> → Mid10<T> → {BranchA<T>, BranchB<T>} → Leaf10<T>
//
// First a chain of depth 2 (Root → Mid), then a diamond on top
// (BranchA, BranchB both extend Mid; Leaf extends both branches).
// Tests that the chain walk correctly handles both linear and
// diamond segments in the same hierarchy.
// =========================================================================

trait Root10<T>: TraitMetadataTable<dyn Root10<T>> + core::fmt::Debug {
    fn root_val(&self) -> u32;
}
trait Mid10<T>: Root10<T> {
    fn mid_val(&self) -> u32;
}
trait BranchA<T>: Mid10<T> {
    fn a_val(&self) -> u32;
}
trait BranchB<T>: Mid10<T> {
    fn b_val(&self) -> u32;
}
trait Leaf10<T>: BranchA<T> + BranchB<T> {
    fn leaf_val(&self) -> u32;
}

#[derive(Debug)]
struct C10<'a> { x: &'a u32 }

impl<'a> Root10<&'a u32> for C10<'a> { fn root_val(&self) -> u32 { *self.x } }
impl<'a> Mid10<&'a u32> for C10<'a> { fn mid_val(&self) -> u32 { *self.x + 1 } }
impl<'a> BranchA<&'a u32> for C10<'a> { fn a_val(&self) -> u32 { *self.x + 2 } }
impl<'a> BranchB<&'a u32> for C10<'a> { fn b_val(&self) -> u32 { *self.x + 3 } }
impl<'a> Leaf10<&'a u32> for C10<'a> { fn leaf_val(&self) -> u32 { *self.x + 4 } }

#[inline(never)]
fn case10<'a>(x: &'a u32) {
    let obj: &dyn Root10<&'_ u32> = &C10 { x };
    let mid = core::cast!(in dyn Root10<&'_ u32>, obj => dyn Mid10<&'_ u32>)
        .expect("case10_mid");
    assert_eq!(mid.mid_val(), *x + 1);
    let a = core::cast!(in dyn Root10<&'_ u32>, obj => dyn BranchA<&'_ u32>)
        .expect("case10_a");
    assert_eq!(a.a_val(), *x + 2);
    let b = core::cast!(in dyn Root10<&'_ u32>, obj => dyn BranchB<&'_ u32>)
        .expect("case10_b");
    assert_eq!(b.b_val(), *x + 3);
    let leaf = core::cast!(in dyn Root10<&'_ u32>, obj => dyn Leaf10<&'_ u32>)
        .expect("case10_leaf");
    assert_eq!(leaf.leaf_val(), *x + 4);
}

// =========================================================================
// Case 11: Multi-type-param diamond
//
// Root11<T, U> → MidT<T, U>, MidU<T, U> → Sub11<T, U>: MidT + MidU
//
// Both type params carry distinct lifetimes via T = &'a u32, U = &'b u32.
// The diamond join must reconcile both type params consistently.
// =========================================================================

trait Root11<T, U>: TraitMetadataTable<dyn Root11<T, U>> + core::fmt::Debug {
    fn t_val(&self) -> u32;
    fn u_val(&self) -> u32;
}
trait MidT<T, U>: Root11<T, U> {
    fn focus_t(&self) -> u32;
}
trait MidU<T, U>: Root11<T, U> {
    fn focus_u(&self) -> u32;
}
trait Sub11<T, U>: MidT<T, U> + MidU<T, U> {
    fn combined(&self) -> u32;
}

#[derive(Debug)]
struct C11<'a, 'b> { t: &'a u32, u: &'b u32 }

impl<'a, 'b> Root11<&'a u32, &'b u32> for C11<'a, 'b> {
    fn t_val(&self) -> u32 { *self.t }
    fn u_val(&self) -> u32 { *self.u }
}
impl<'a, 'b> MidT<&'a u32, &'b u32> for C11<'a, 'b> {
    fn focus_t(&self) -> u32 { *self.t * 2 }
}
impl<'a, 'b> MidU<&'a u32, &'b u32> for C11<'a, 'b> {
    fn focus_u(&self) -> u32 { *self.u * 3 }
}
impl<'a, 'b> Sub11<&'a u32, &'b u32> for C11<'a, 'b> {
    fn combined(&self) -> u32 { *self.t + *self.u }
}

#[inline(never)]
fn case11<'a>(x: &'a u32) {
    let obj: &dyn Root11<&'_ u32, &'_ u32> = &C11 { t: x, u: x };
    let mt = core::cast!(
        in dyn Root11<&'_ u32, &'_ u32>,
        obj => dyn MidT<&'_ u32, &'_ u32>
    ).expect("case11_t");
    assert_eq!(mt.focus_t(), *x * 2);
    let mu = core::cast!(
        in dyn Root11<&'_ u32, &'_ u32>,
        obj => dyn MidU<&'_ u32, &'_ u32>
    ).expect("case11_u");
    assert_eq!(mu.focus_u(), *x * 3);
    let sub = core::cast!(
        in dyn Root11<&'_ u32, &'_ u32>,
        obj => dyn Sub11<&'_ u32, &'_ u32>
    ).expect("case11_sub");
    assert_eq!(sub.combined(), *x + *x);
}

fn main() {
    let x: u32 = 10;

    eprintln!("=== case1: basic ref type param ===");
    case1(&x);

    eprintln!("=== case2: two type params merged ===");
    case2(&x);

    eprintln!("=== case3: where-clause gating ===");
    case3(&x);

    eprintln!("=== case4: Option type param ===");
    case4_some(&x);
    case4_none(&x);

    eprintln!("=== case5: chain depth-3 ===");
    case5(&x);

    eprintln!("=== case6: diamond ===");
    case6(&x);

    eprintln!("=== case7: mixed lifetime + type params ===");
    case7(&x);

    eprintln!("=== case8: type param + projected assoc ===");
    case8(&x);

    eprintln!("=== case9: type param + projection ===");
    case9(&x);

    eprintln!("=== case10: chain + diamond ===");
    case10(&x);

    eprintln!("=== case11: multi-type-param diamond ===");
    case11(&x);
}
