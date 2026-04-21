//! Diagnostic (eager): the `TraitMetadataTable<T>` type argument must be a
//! trait object. Non-`dyn` arguments render the bound uninhabitable and are
//! never what the author intended.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

trait ChildTrait: TraitMetadataTable<u32> {}
//~^ ERROR `TraitMetadataTable` type argument must be a trait object
//~| ERROR E0271
