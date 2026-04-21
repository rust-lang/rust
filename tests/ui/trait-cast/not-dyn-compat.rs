//! Negative compile-fail tests: trait patterns that are NOT dyn-compatible
//! (object-safe) and therefore cannot participate in trait-cast.
//!
//! Trait-cast requires `TraitMetadataTable<dyn Trait>` as a supertrait
//! bound, which forces dyn compatibility on the trait.  Any feature
//! that breaks dyn compatibility — generic methods, `Self: Sized`
//! requirements, methods returning `Self`, generic methods using
//! associated types — must be rejected.
//!
//! Each case mirrors a positive pattern from lifetime-in-generics.rs
//! but introduces a feature that makes the trait non-dyn-compat.
//!
//! Note: GAT cases (`type Lent<'a>`) trigger a compiler ICE in
//! `check_well_formed` rather than a clean dyn-compatibility error, so
//! they cannot be expressed as ordinary compile-fail UI tests here.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

// =========================================================================
// Negative 1: Method with generic type parameter — not dyn-compat
//
// `process<U>` is a generic method.  Generic methods cannot be
// dispatched through a vtable.  Mirrors the positive Root1<T> pattern
// (case 1) — replacing the trait-level type param with a method-level
// one is invalid.
// =========================================================================

trait GenericMethod<T>: TraitMetadataTable<dyn GenericMethod<T>> + core::fmt::Debug {
//~^ ERROR the trait `GenericMethod` is not dyn compatible
//~| ERROR the trait `GenericMethod` is not dyn compatible
    fn process<U>(&self, value: U) -> u32;
}

// =========================================================================
// Negative 2: `Self: Sized` requirement on the trait — not dyn-compat
//
// Requiring `Self: Sized` excludes dyn types (which are unsized).
// Mirrors the positive single-type-param pattern (case 1).
// =========================================================================

trait NeedsSized<T>: TraitMetadataTable<dyn NeedsSized<T>> + core::fmt::Debug
//~^ ERROR the trait `NeedsSized` is not dyn compatible
//~| ERROR the trait `NeedsSized` is not dyn compatible
where
    Self: Sized,
{
    fn val(&self) -> u32;
}

// =========================================================================
// Negative 3: Method returning Self — not dyn-compat
//
// Returning `Self` requires sizing the return value.  Mirrors the
// diamond pattern (case 6) — but with a builder-style method.
// =========================================================================

trait ReturnsSelf<T>: TraitMetadataTable<dyn ReturnsSelf<T>> + core::fmt::Debug {
//~^ ERROR the trait `ReturnsSelf` is not dyn compatible
//~| ERROR the trait `ReturnsSelf` is not dyn compatible
//~| ERROR the trait `ReturnsSelf` is not dyn compatible
    fn val(&self) -> u32;
    fn build(&self) -> Self;
}

// =========================================================================
// Negative 4: Method with generic type param using associated type — not
// dyn-compat
//
// Combines the associated-type pattern (positive case 8) with a
// generic method.  Even though associated types alone are dyn-compat,
// the generic method `map<F>` breaks it.
// =========================================================================

trait AssocGeneric<T>:
    TraitMetadataTable<dyn AssocGeneric<T, Assoc = T>> + core::fmt::Debug
//~^ ERROR the trait `AssocGeneric` is not dyn compatible
//~| ERROR the trait `AssocGeneric` is not dyn compatible
{
    type Assoc;
    fn map<F: Fn(&Self::Assoc) -> u32>(&self, f: F) -> u32;
}
