//! One of the two global cdylibs used by this test.
//!
//! Exposes:
//!
//!   * `cdylib_a_make_root`      — returns a `&'static dyn Root` wrapped
//!                                 in a `RootRef` (leaks a static `TypeA`).
//!   * `cdylib_a_cast_sub`       — runs `core::cast!(in dyn Root, _ => dyn Sub)`
//!                                 on the incoming carrier and reports the
//!                                 outcome via status codes.
//!
//! The trait-metadata intrinsics in `cdylib_a_cast_sub` are monomorphized
//! *here* (since this crate is a global crate), so they return this
//! cdylib's `global_crate_id`. When the bin hands in a carrier produced
//! by `cdylib_b` (or from a bin-local type), the ids diverge and
//! `TraitCastError::ForeignTraitGraph` is raised.

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![crate_type = "cdylib"]

extern crate common;
extern crate core;

use core::trait_cast::TraitCastError;

use common::{CAST_FOREIGN_GRAPH, CAST_OK, CAST_UNSATISFIED, Root, RootRef, Sub};

pub struct TypeA;

impl Root for TypeA {
    fn name(&self) -> &'static str {
        "TypeA"
    }
}

impl Sub for TypeA {
    fn sub_value(&self) -> u32 {
        0xA
    }
}

static TYPE_A: TypeA = TypeA;

#[no_mangle]
pub extern "C" fn cdylib_a_make_root() -> RootRef {
    RootRef::from_root(&TYPE_A as &dyn Root)
}

/// # Safety
/// `obj` must be a valid `RootRef` pointing at a live `dyn Root`.
#[no_mangle]
pub unsafe extern "C" fn cdylib_a_cast_sub(obj: RootRef, out_value: *mut u32) -> i32 {
    let r: &dyn Root = unsafe { obj.as_root() };
    match core::try_cast!(in dyn Root, r => dyn Sub) {
        Ok(s) => {
            unsafe { *out_value = s.sub_value() };
            CAST_OK
        }
        Err(TraitCastError::ForeignTraitGraph(_)) => CAST_FOREIGN_GRAPH,
        Err(TraitCastError::UnsatisfiedObligation(_)) => CAST_UNSATISFIED,
    }
}
