//! Diagnostic (eager): a trait-cast sub-trait introduces a lifetime
//! parameter that the root supertrait cannot bound. Such a lifetime is
//! erased on unsizing to `dyn Root` and would be manufactured at
//! downcast time — unsound.
//!
//! Emitted at trait-definition time, not at cast sites, so the error
//! surfaces even when no `cast!` is written in this crate.

#![feature(trait_cast)]
#![allow(dead_code)]

#![crate_type = "rlib"]

extern crate core;
use core::marker::TraitMetadataTable;

// Root with no lifetime parameters — accepts only subtraits that don't
// introduce lifetime parameters of their own.
trait Root: TraitMetadataTable<dyn Root> + core::fmt::Debug {
    fn val(&self) -> u32;
}

// Sub-trait introducing `'a` which cannot be expressed through `Root`.
trait Sub<'a>: Root {
    //~^ ERROR trait graph rooted at `Root` is not downcast-safe
    fn f(&self) -> &'a u8;
}
