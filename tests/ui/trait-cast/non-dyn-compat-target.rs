//! Diagnostic: non-dyn-compatible cast target.
//!
//! When a `cast!` expression names a non-dyn-compatible trait as its
//! target, lowering `dyn Target` triggers the existing E0038
//! dyn-compatibility diagnostic before any trait-cast-specific
//! machinery runs.  This test verifies that the error path surfaces
//! cleanly through the macro expansion.
//!
//! The root (`CleanRoot`) is itself dyn-compatible; only the target
//! (`NotDynCompat`) is rejected.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

// A dyn-compatible root.
trait CleanRoot: TraitMetadataTable<dyn CleanRoot> + core::fmt::Debug {
    fn val(&self) -> u32;
}

// A sub-trait that breaks dyn-compatibility by returning `Self`.
trait NotDynCompat: CleanRoot {
    fn build(&self) -> Self;
}

#[derive(Debug)]
struct S;

impl CleanRoot for S {
    fn val(&self) -> u32 { 0 }
}

fn cast_to_non_dyn_compat(obj: &dyn CleanRoot) {
    let _ = core::cast!(in dyn CleanRoot, obj => dyn NotDynCompat);
    //~^ ERROR E0038
}

fn main() {}
