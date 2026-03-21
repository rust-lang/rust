//! Diagnostic (eager): `TraitMetadataTable<dyn X>` supertrait bound where
//! `dyn X` is neither `dyn Self` (declaring this trait as a cast root) nor
//! `dyn R` for a transitive supertrait `R` that is itself a cast root.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

// A valid cast root.
trait Root: TraitMetadataTable<dyn Root> + core::fmt::Debug {
    fn val(&self) -> u32;
}

// A separate, unrelated cast root.
trait Unrelated: TraitMetadataTable<dyn Unrelated> + core::fmt::Debug {
    fn other(&self) -> u32;
}

// Sub-trait that points its `TraitMetadataTable` at an unrelated root,
// rather than `Self` or `Root` (its own supertrait).
trait ChildTrait: Root + TraitMetadataTable<dyn Unrelated> {}
//~^ ERROR `TraitMetadataTable` type argument does not match a cast root
