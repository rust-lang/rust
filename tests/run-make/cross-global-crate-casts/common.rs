//! Shared rlib used by `cdylib_a`, `cdylib_b`, and `program`.
//!
//! This crate is *not* a global crate (it's an rlib), so it defines no
//! trait metadata tables of its own. Every downstream global crate
//! (the two cdylibs and the bin) computes its own table over the types
//! it can see — deliberately giving three disjoint global crates that
//! all assign the same slot index to `dyn Sub` but hand out different
//! `global_crate_id` tokens. That divergence is what the runtime cast
//! check distinguishes.

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![feature(ptr_metadata)]
#![crate_type = "rlib"]

extern crate core;

use core::marker::TraitMetadataTable;
use core::ptr::{DynMetadata, Pointee};

pub trait Root: TraitMetadataTable<dyn Root> {
    fn name(&self) -> &'static str;
}

pub trait Sub: Root {
    fn sub_value(&self) -> u32;
}

/// FFI-safe carrier for `*const dyn Root`.
///
/// The bin and the two cdylibs can't name each other's concrete types,
/// so they exchange erased trait objects across the extern-"C" boundary
/// via this struct. Layout: `(data ptr, vtable ptr)` — identical to
/// `*const dyn Root`, but explicitly `#[repr(C)]` so we don't rely on
/// the implicit layout of a `*const dyn T`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RootRef {
    pub data: *const (),
    pub vtable: *const (),
}

impl RootRef {
    pub fn from_root(r: &dyn Root) -> Self {
        let raw: *const dyn Root = r;
        let (data, meta) = raw.to_raw_parts();
        let vtable: *const () = unsafe { core::mem::transmute(meta) };
        RootRef { data, vtable }
    }

    /// # Safety
    /// The carrier must have been produced by `RootRef::from_root` in
    /// some (possibly different) global crate and the pointee must still
    /// be live.
    pub unsafe fn as_root<'a>(self) -> &'a dyn Root {
        let meta: <dyn Root as Pointee>::Metadata =
            unsafe { core::mem::transmute::<*const (), DynMetadata<dyn Root>>(self.vtable) };
        let ptr: *const dyn Root = core::ptr::from_raw_parts(self.data, meta);
        unsafe { &*ptr }
    }
}

/// Status codes used by the extern-"C" cast entry points. Kept in the
/// shared crate so the bin and cdylibs agree.
pub const CAST_OK: i32 = 0;
pub const CAST_FOREIGN_GRAPH: i32 = 1;
pub const CAST_UNSATISFIED: i32 = 2;
