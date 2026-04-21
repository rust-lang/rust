//! Driver for the cross-global-crate cast test.
//!
//! This bin is itself a global crate (CrateType::Executable). It links
//! against `cdylib_a` and `cdylib_b` purely through their extern-"C"
//! entry points — the bin never names `TypeA`/`TypeB` and the cdylibs
//! never name `BinType`.
//!
//! There are therefore three independent global crates at runtime:
//!
//!   * the bin,
//!   * `libcdylib_a.so`,
//!   * `libcdylib_b.so`.
//!
//! Each owns a distinct `global_crate_id` allocation. The RFC guarantees
//! that trait metadata index + table checks compare crate-id tokens
//! before trusting slot indices, so even though all three graphs happen
//! to assign slot 0 to `dyn Sub`, casts across crate boundaries must
//! report `TraitCastError::ForeignTraitGraph`. Casts that stay inside
//! one global crate must succeed.
//!
//! The test matrix covered by `main` below:
//!
//!                      cast site
//!              ┌────────────┬────────────┬────────────┐
//!   object  ── │  bin       │  cdylib_a  │  cdylib_b  │
//! ─────────────┼────────────┼────────────┼────────────┤
//!   BinType    │  ok        │  foreign   │  foreign   │
//!   TypeA      │  foreign   │  ok        │  foreign   │
//!   TypeB      │  foreign   │  foreign   │  ok        │
//!              └────────────┴────────────┴────────────┘

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![crate_type = "bin"]

extern crate common;
extern crate core;

use core::trait_cast::TraitCastError;

use common::{CAST_FOREIGN_GRAPH, CAST_OK, Root, RootRef, Sub};

// ---- extern surface of the two cdylibs --------------------------------

#[link(name = "cdylib_a")]
unsafe extern "C" {
    fn cdylib_a_make_root() -> RootRef;
    fn cdylib_a_cast_sub(obj: RootRef, out_value: *mut u32) -> i32;
}

#[link(name = "cdylib_b")]
unsafe extern "C" {
    fn cdylib_b_make_root() -> RootRef;
    fn cdylib_b_cast_sub(obj: RootRef, out_value: *mut u32) -> i32;
}

// ---- bin-local trait impl --------------------------------------------

struct BinType;

impl Root for BinType {
    fn name(&self) -> &'static str {
        "BinType"
    }
}

impl Sub for BinType {
    fn sub_value(&self) -> u32 {
        0xB1
    }
}

// ---- cast helpers ----------------------------------------------------

/// Run the cast at the bin's cast site and classify the outcome the same
/// way the cdylibs do — this keeps the three rows of the matrix above
/// mutually comparable.
fn bin_cast_sub(obj: RootRef) -> (i32, u32) {
    let r: &dyn Root = unsafe { obj.as_root() };
    let mut value = 0u32;
    let code = match core::try_cast!(in dyn Root, r => dyn Sub) {
        Ok(s) => {
            value = s.sub_value();
            CAST_OK
        }
        Err(TraitCastError::ForeignTraitGraph(_)) => CAST_FOREIGN_GRAPH,
        Err(TraitCastError::UnsatisfiedObligation(_)) => common::CAST_UNSATISFIED,
    };
    (code, value)
}

fn a_cast_sub(obj: RootRef) -> (i32, u32) {
    let mut value = 0u32;
    let code = unsafe { cdylib_a_cast_sub(obj, &mut value) };
    (code, value)
}

fn b_cast_sub(obj: RootRef) -> (i32, u32) {
    let mut value = 0u32;
    let code = unsafe { cdylib_b_cast_sub(obj, &mut value) };
    (code, value)
}

fn main() {
    let a_obj: RootRef = unsafe { cdylib_a_make_root() };
    let b_obj: RootRef = unsafe { cdylib_b_make_root() };
    let bin_type = BinType;
    let bin_obj: RootRef = RootRef::from_root(&bin_type as &dyn Root);

    // Sanity: vtable dispatch across the FFI boundary still works for
    // the non-cast method — if this failed, the test below would be
    // diagnosing something unrelated.
    assert_eq!(unsafe { a_obj.as_root() }.name(), "TypeA");
    assert_eq!(unsafe { b_obj.as_root() }.name(), "TypeB");
    assert_eq!(unsafe { bin_obj.as_root() }.name(), "BinType");

    // ---- diagonal: cast site matches object's origin global crate ----

    assert_eq!(bin_cast_sub(bin_obj), (CAST_OK, 0xB1));
    assert_eq!(a_cast_sub(a_obj), (CAST_OK, 0xA));
    assert_eq!(b_cast_sub(b_obj), (CAST_OK, 0xB));

    // ---- off-diagonal: every cross-crate pair must reject ------------

    // bin cast site, foreign-origin objects.
    assert_eq!(bin_cast_sub(a_obj), (CAST_FOREIGN_GRAPH, 0));
    assert_eq!(bin_cast_sub(b_obj), (CAST_FOREIGN_GRAPH, 0));

    // cdylib_a cast site, foreign-origin objects.
    assert_eq!(a_cast_sub(bin_obj), (CAST_FOREIGN_GRAPH, 0));
    assert_eq!(a_cast_sub(b_obj), (CAST_FOREIGN_GRAPH, 0));

    // cdylib_b cast site, foreign-origin objects.
    assert_eq!(b_cast_sub(bin_obj), (CAST_FOREIGN_GRAPH, 0));
    assert_eq!(b_cast_sub(a_obj), (CAST_FOREIGN_GRAPH, 0));
}
