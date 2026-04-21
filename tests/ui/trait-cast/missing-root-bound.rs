//! Diagnostic: trait used as a `cast!` root lacks `TraitMetadataTable<dyn Self>`.
//!
//! The `cast!(in dyn Root, ...)` macro requires that `Root` carry
//! `TraitMetadataTable<dyn Root>` as a supertrait bound. When it
//! does not, the specialized `MissingRootBound` diagnostic is emitted
//! instead of the generic "trait bound not satisfied" error.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;

// A trait that does NOT carry `TraitMetadataTable<dyn BrokenRoot>`.
trait BrokenRoot: core::fmt::Debug {
    fn val(&self) -> u32;
}

// A hypothetical target.  Even if it had `TraitMetadataTable<dyn BrokenRoot>`
// as a supertrait, the root itself would still fail the check — this fixture
// focuses on the root-missing case.
trait SomeTarget: BrokenRoot {
    fn extra(&self) -> u32;
}

#[derive(Debug)]
struct S;

impl BrokenRoot for S {
    fn val(&self) -> u32 { 0 }
}

fn cast_through_broken_root(obj: &dyn BrokenRoot) {
    let _ = core::cast!(in dyn BrokenRoot, obj => dyn SomeTarget);
    //~^ ERROR `BrokenRoot` cannot be used as a cast root
    //~| ERROR `BrokenRoot` cannot be used as a cast root
    //~| ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
}
