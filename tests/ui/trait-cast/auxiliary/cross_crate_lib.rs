//! Upstream (non-global) library crate used by
//! `../cross-crate-casts.rs`.  Rlib default — not a global crate, so
//! every trait-cast intrinsic emitted here must flow through the
//! `DelayedInstance` rmeta round-trip to be resolved downstream.

#![feature(trait_cast)]
#![feature(sized_hierarchy)]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

pub trait Root: TraitMetadataTable<dyn Root> + core::fmt::Debug {
    fn name(&self) -> &'static str;
}

pub trait Greet: Root {
    fn greeting(&self) -> &'static str;
}

pub trait Count: Root {
    fn count(&self) -> u32;
}

pub trait Describe: Root {
    fn description(&self) -> &'static str;
}

// ---- upstream concrete types ----

#[derive(Debug)]
pub struct LibTypeA;

#[derive(Debug)]
pub struct LibTypeB;

impl Root for LibTypeA {
    fn name(&self) -> &'static str { "LibTypeA" }
}
impl Greet for LibTypeA {
    fn greeting(&self) -> &'static str { "hello from LibTypeA" }
}
impl Count for LibTypeA {
    fn count(&self) -> u32 { 1 }
}
impl Describe for LibTypeA {
    fn description(&self) -> &'static str { "the describable one" }
}

impl Root for LibTypeB {
    fn name(&self) -> &'static str { "LibTypeB" }
}
impl Greet for LibTypeB {
    fn greeting(&self) -> &'static str { "hello from LibTypeB" }
}
impl Count for LibTypeB {
    fn count(&self) -> u32 { 2 }
}
// NOTE: no Describe impl for LibTypeB — the corresponding downstream
// cast must return Err at runtime.

// ---- upstream cast sites ----

/// Non-generic upstream cast site.  The call to `core::cast!` expands to
/// `trait_metadata_index<dyn Root, dyn Describe>` +
/// `trait_metadata_table<dyn Root, _>` intrinsics.  Both indices are
/// unknown until the downstream global crate runs `trait_cast_layout`,
/// so this function is recorded in the upstream crate's
/// `delayed_codegen_requests` and its final codegen happens in the
/// global crate.
///
/// `#[inline]` forces `cross_crate_inlinable`, which forces the
/// encoder to ship this fn's optimized MIR in rmeta.  Without the
/// annotation, `should_encode_mir` skips non-generic, non-inline fns
/// and the downstream mono collector ICEs trying to decode the body
/// at `cascade_canonicalize` time — a real delayed-codegen bug: the
/// `has_trait_cast_intrinsics` query only scans the direct body, so
/// functions whose intrinsic calls only materialize via
/// post-monomorphization inlining of `TraitCast::checked_cast` (from
/// `core`) are not detected as needing cross-crate MIR.
#[inline]
pub fn try_describe_from_lib(obj: &dyn Root) -> Option<&'static str> {
    core::cast!(in dyn Root, obj => dyn Describe)
        .ok()
        .map(|d| d.description())
}

/// Upstream factory returning a boxed trait object.  The unsizing
/// `LibTypeA -> dyn Root` produces a vtable whose
/// `TraitMetadataTable::derived_metadata_table` slot points at the
/// blanket-impl monomorphization of `trait_metadata_table<dyn Root,
/// LibTypeA>` — itself a delayed intrinsic.
pub fn lib_boxed_a() -> Box<dyn Root> {
    Box::new(LibTypeA)
}

pub fn lib_boxed_b() -> Box<dyn Root> {
    Box::new(LibTypeB)
}
