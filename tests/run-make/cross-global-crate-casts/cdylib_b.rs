//! Second global cdylib. Structurally identical to `cdylib_a`, but with
//! a distinct concrete type (`TypeB`) and a distinct `global_crate_id`
//! allocation. The interesting property is that `cdylib_b`'s
//! `trait_metadata_index<dyn Root, dyn Sub>` picks the *same* slot
//! index as `cdylib_a`'s (both graphs contain only `dyn Sub` as a
//! sub-trait of `dyn Root`). Only the crate-id token distinguishes
//! them, and that's what the test exercises.

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![crate_type = "cdylib"]

extern crate common;
extern crate core;

use core::trait_cast::TraitCastError;

use common::{CAST_FOREIGN_GRAPH, CAST_OK, CAST_UNSATISFIED, Root, RootRef, Sub};

pub struct TypeB;

impl Root for TypeB {
    fn name(&self) -> &'static str {
        "TypeB"
    }
}

impl Sub for TypeB {
    fn sub_value(&self) -> u32 {
        0xB
    }
}

static TYPE_B: TypeB = TypeB;

#[no_mangle]
pub extern "C" fn cdylib_b_make_root() -> RootRef {
    RootRef::from_root(&TYPE_B as &dyn Root)
}

/// # Safety
/// `obj` must be a valid `RootRef` pointing at a live `dyn Root`.
#[no_mangle]
pub unsafe extern "C" fn cdylib_b_cast_sub(obj: RootRef, out_value: *mut u32) -> i32 {
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
