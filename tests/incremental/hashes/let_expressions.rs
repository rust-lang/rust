// This test case tests the incremental compilation hash (ICH) implementation
// for let expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ revisions: bpass1 bpass2 bpass3 bpass4 bpass5 bpass6
//@ compile-flags: -Z query-dep-graph -O
//@ [bpass1]compile-flags: -Zincremental-ignore-spans
//@ [bpass2]compile-flags: -Zincremental-ignore-spans
//@ [bpass3]compile-flags: -Zincremental-ignore-spans
//@ ignore-backends: gcc
// FIXME(#62277): could be check-pass?

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Change Name -----------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_name() {
    let _x = 2u64;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_name() {
    let _y = 2u64;
}



// Add Type --------------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_type() {
    let _x      = 2u32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_type() {
    let _x: u32 = 2u32;
}



// Change Type -----------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_type() {
    let _x: u64 = 2;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_type() {
    let _x: u8  = 2;
}



// Change Mutability of Reference Type -----------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_mutability_of_reference_type() {
    let _x: &    u64;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_mutability_of_reference_type() {
    let _x: &mut u64;
}



// Change Mutability of Slot ---------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_mutability_of_slot() {
    let mut _x: u64 = 0;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_mutability_of_slot() {
    let     _x: u64 = 0;
}



// Change Simple Binding to Pattern --------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_simple_binding_to_pattern() {
    let  _x      = (0u8, 'x');
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_simple_binding_to_pattern() {
    let (_a, _b) = (0u8, 'x');
}



// Change Name in Pattern ------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_name_in_pattern() {
    let (_a, _b) = (1u8, 'y');
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_name_in_pattern() {
    let (_a, _c) = (1u8, 'y');
}



// Add `ref` in Pattern --------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_ref_in_pattern() {
    let (    _a, _b) = (1u8, 'y');
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn add_ref_in_pattern() {
    let (ref _a, _b) = (1u8, 'y');
}



// Add `&` in Pattern ----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_amp_in_pattern() {
    let ( _a, _b) = (&1u8, 'y');
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn add_amp_in_pattern() {
    let (&_a, _b) = (&1u8, 'y');
}



// Change Mutability of Binding in Pattern -------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_mutability_of_binding_in_pattern() {
    let (    _a, _b) = (99u8, 'q');
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_mutability_of_binding_in_pattern() {
    let (mut _a, _b) = (99u8, 'q');
}



// Add Initializer -------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_initializer() {
    let _x: i16       ;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn add_initializer() {
    let _x: i16 = 3i16;
}



// Change Initializer ----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_initializer() {
    let _x = 4u16;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_initializer() {
    let _x = 5u16;
}
