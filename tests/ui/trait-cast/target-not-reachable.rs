//! Diagnostic: cast target trait is not reachable from the root supertrait.
//!
//! When `cast!(in dyn Root, obj => dyn Target)` names a target whose
//! principal trait is not a (transitive) supertrait of `Root`, the
//! specialized `TargetNotReachable` diagnostic is emitted instead of
//! the generic "trait bound not satisfied" error.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

// Graph A: Root with its own sub-trait.
trait Root: TraitMetadataTable<dyn Root> + core::fmt::Debug {
    fn val(&self) -> u32;
}

// Completely unrelated trait — has its own metadata table, but no shared
// root with `Root`.
trait Unrelated: TraitMetadataTable<dyn Unrelated> + core::fmt::Debug {
    fn do_something(&self);
}

#[derive(Debug)]
struct S;

impl Root for S {
    fn val(&self) -> u32 { 0 }
}

fn target_not_in_graph(obj: &dyn Root) {
    let _ = core::cast!(in dyn Root, obj => dyn Unrelated);
    //~^ ERROR `Unrelated` is not in the trait graph rooted at `Root`
    //~| ERROR `Unrelated` is not in the trait graph rooted at `Root`
    //~| ERROR E0277
}
