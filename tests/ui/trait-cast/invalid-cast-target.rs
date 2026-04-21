//! Negative compile-fail tests: cast! invocations whose TARGET cannot
//! be reached from the source dyn type within the trait graph.
//!
//! Existing compile-fail coverage (not-dyn-compat.rs) tests traits
//! whose shape breaks dyn compatibility.  This file complements that
//! by testing casts to dyn types that ARE individually dyn-compatible
//! but are not reachable from the source's trait graph.  Each target
//! fails `E0277` because `TraitCast<I, U>` requires
//! `U: TraitMetadataTable<I>`, and the auto-impled
//! `TraitMetadataTable` chain does not reach the target from the
//! source's trait graph:
//!
//!   1. Sub-trait with a wrong generic type parameter.
//!   2. Sub-trait of an entirely different root trait.
//!   3. A trait that is not a sub-trait of anything (no connection).
//!
//! Note: a naive wrong-projection target
//! (e.g. `dyn Sub<T, Assoc = u64>` when the source carries
//! `Assoc = &'a u32`) does NOT fail compile-time; the auto-impled
//! `TraitMetadataTable` accepts the mismatched projection.  Such
//! mismatches surface as a *runtime* erasure-safety failure and are
//! covered by `erasure-safety-projections.rs` / `runtime-cast-failures.rs`
//! rather than here.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Shared Root1 / Sub1 from the positive `lifetime-in-generics` case 1.
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

// =========================================================================
// Invalid 1: Sub-trait with the wrong generic type parameter.
//
// Cast target is `dyn Sub1<u64>` but the source holds `&'_ u32`.  The
// auto-impled `TraitMetadataTable<dyn Root1<T>>` for `dyn Sub1<T>`
// uses the same `T` on both sides; substituting `u64` for the target
// breaks the chain to `dyn Root1<&u32>`.
// =========================================================================

fn invalid1<'a>(x: &'a u32) {
    let obj: &dyn Root1<&'_ u32> = &S1 { x };
    let _ = core::cast!(in dyn Root1<&'_ u32>, obj => dyn Sub1<u64>);
    //~^ ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
}

// =========================================================================
// Invalid 2: Sub-trait of a DIFFERENT root trait.
//
// Root2 is a separate trait graph with its own sub-trait Sub2.
// Casting from `dyn Root1<_>` to `dyn Sub2<_>` fails because Sub2's
// `TraitMetadataTable` chain connects it to Root2, not Root1.
// =========================================================================

trait Root2<T>: TraitMetadataTable<dyn Root2<T>> + core::fmt::Debug {
    fn val(&self) -> u32;
}
trait Sub2<T>: Root2<T> {
    fn sub_val(&self) -> u32;
}

fn invalid2<'a>(x: &'a u32) {
    let obj: &dyn Root1<&'_ u32> = &S1 { x };
    let _ = core::cast!(in dyn Root1<&'_ u32>, obj => dyn Sub2<&'_ u32>);
    //~^ ERROR `Sub2` is not in the trait graph rooted at `Root1`
    //~| ERROR `Sub2` is not in the trait graph rooted at `Root1`
    //~| ERROR E0277
}

// =========================================================================
// Invalid 3: Completely unrelated dyn trait (no root relationship).
//
// `Unrelated` carries its own `TraitMetadataTable<dyn Unrelated>`
// supertrait but is not a sub-trait of any shared root with Root1.
// =========================================================================

trait Unrelated: TraitMetadataTable<dyn Unrelated> + core::fmt::Debug {
    fn do_something(&self);
}

fn invalid3<'a>(x: &'a u32) {
    let obj: &dyn Root1<&'_ u32> = &S1 { x };
    let _ = core::cast!(in dyn Root1<&'_ u32>, obj => dyn Unrelated);
    //~^ ERROR `Unrelated` is not in the trait graph rooted at `Root1`
    //~| ERROR `Unrelated` is not in the trait graph rooted at `Root1`
    //~| ERROR E0277
}
