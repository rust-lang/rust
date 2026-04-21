//! Condensation baseline: lifetime-parameterized trait hierarchy with
//! multiple call contexts. Exercises the `condense_outlives_classes`
//! code path driven by shared Self/trait params in
//! `impl_admissible_under_class`.
//!
//! Every impl has the form `impl<'a, 'b> SubX<'a, 'b> for TypeA<'a, 'b>`
//! — the lifetime parameters `'a` and `'b` appear in BOTH the impl's
//! Self type AND its trait-ref args. That shared-param condition causes
//! `impl_universally_admissible` to return false for every impl, so
//! `trait_cast_layout` BYPASSES the fast path and invokes
//! `condense_outlives_classes` for every sub-trait.
//!
//! With a single (empty) outlives class materialized per sub-trait,
//! condensation produces exactly one group per sub-trait, yielding
//! `table_length == num_sub_traits`. This baseline captures the
//! minimum-size-via-condensation-path contract: if condensation
//! regressed and started emitting multiple slots for a single class,
//! the table length would grow and this test would fail.
//!
//! Trait graph:
//!   root:         dyn Root<'a, 'b>
//!   sub-traits:   dyn SubX<'a, 'b>, dyn SubY<'a, 'b>
//!   concrete:     TypeA<'a, 'b>, TypeB<'a, 'b>
//!
//! Expected resolutions in post-mono MIR:
//!   trait_metadata_table_len<dyn Root<'_,'_>>()              → 2_usize
//!   trait_metadata_index<dyn Root<'_,'_>, dyn SubX<'_,'_>>   → (.., 0_usize)
//!   trait_metadata_index<dyn Root<'_,'_>, dyn SubY<'_,'_>>   → (.., 1_usize)

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait Root<'a, 'b>: TraitMetadataTable<dyn Root<'a, 'b>> + core::fmt::Debug {
    fn id(&self) -> u32;
}

trait SubX<'a, 'b>: Root<'a, 'b> {
    fn x_val(&self) -> u32;
}

trait SubY<'a, 'b>: Root<'a, 'b> {
    fn y_val(&self) -> u32;
}

// ---- concrete types ----

#[derive(Debug)]
struct TypeA<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

#[derive(Debug)]
struct TypeB<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

// All impls are universally admissible: no where clauses, no param
// aliasing in the trait ref, no shared Self/trait params.
impl<'a, 'b> Root<'a, 'b> for TypeA<'a, 'b> {
    fn id(&self) -> u32 {
        1
    }
}
impl<'a, 'b> Root<'a, 'b> for TypeB<'a, 'b> {
    fn id(&self) -> u32 {
        2
    }
}

impl<'a, 'b> SubX<'a, 'b> for TypeA<'a, 'b> {
    fn x_val(&self) -> u32 {
        *self.x
    }
}
impl<'a, 'b> SubX<'a, 'b> for TypeB<'a, 'b> {
    fn x_val(&self) -> u32 {
        *self.x * 10
    }
}

impl<'a, 'b> SubY<'a, 'b> for TypeA<'a, 'b> {
    fn y_val(&self) -> u32 {
        *self.y
    }
}
impl<'a, 'b> SubY<'a, 'b> for TypeB<'a, 'b> {
    fn y_val(&self) -> u32 {
        *self.y * 10
    }
}

// ---- multiple coercion/cast contexts ----

/// Context 1: both lifetimes equal (single outer lifetime).
#[inline(never)]
fn ctx_equal<'a>(x: &'a u32, y: &'a u32) {
    let a = TypeA { x, y };
    let obj: &dyn Root<'_, '_> = &a;
    let sx = core::cast!(in dyn Root<'_, '_>, obj => dyn SubX<'_, '_>).expect("ctx_equal: subx");
    assert_eq!(sx.x_val(), *x);
    let sy = core::cast!(in dyn Root<'_, '_>, obj => dyn SubY<'_, '_>).expect("ctx_equal: suby");
    assert_eq!(sy.y_val(), *y);
}

/// Context 2: `'b` is strictly interior to `'a`.
#[inline(never)]
fn ctx_interior<'a>(x: &'a u32) {
    let local: u32 = 7;
    let b = TypeB { x, y: &local };
    let obj: &dyn Root<'_, '_> = &b;
    let sx = core::cast!(in dyn Root<'_, '_>, obj => dyn SubX<'_, '_>).expect("ctx_interior: subx");
    assert_eq!(sx.x_val(), *x * 10);
    let sy = core::cast!(in dyn Root<'_, '_>, obj => dyn SubY<'_, '_>).expect("ctx_interior: suby");
    assert_eq!(sy.y_val(), 7 * 10);
}

/// Context 3: explicit `'a: 'b` bound introduced in the signature.
#[inline(never)]
fn ctx_bounded<'a, 'b>(x: &'a u32, y: &'b u32)
where
    'a: 'b,
{
    let a = TypeA { x, y };
    let obj: &dyn Root<'_, '_> = &a;
    let sx = core::cast!(in dyn Root<'_, '_>, obj => dyn SubX<'_, '_>).expect("ctx_bounded: subx");
    assert_eq!(sx.x_val(), *x);
    let sy = core::cast!(in dyn Root<'_, '_>, obj => dyn SubY<'_, '_>).expect("ctx_bounded: suby");
    assert_eq!(sy.y_val(), *y);
}

/// Context 4: 'static everywhere.
#[inline(never)]
fn ctx_static() {
    static X: u32 = 100;
    static Y: u32 = 200;
    let b = TypeB { x: &X, y: &Y };
    let obj: &dyn Root<'static, 'static> = &b;
    let sx = core::cast!(in dyn Root<'static, 'static>, obj => dyn SubX<'static, 'static>)
        .expect("ctx_static: subx");
    assert_eq!(sx.x_val(), 100 * 10);
    let sy = core::cast!(in dyn Root<'static, 'static>, obj => dyn SubY<'static, 'static>)
        .expect("ctx_static: suby");
    assert_eq!(sy.y_val(), 200 * 10);
}

fn main() {
    let a: u32 = 3;
    let b: u32 = 5;
    ctx_equal(&a, &b);
    ctx_interior(&a);
    ctx_bounded(&a, &b);
    ctx_static();
}
