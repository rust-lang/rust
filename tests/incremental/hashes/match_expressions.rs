// This test case tests the incremental compilation hash (ICH) implementation
// for match expressions.

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

// Add Arm ---------------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        /*---*/
        _ => 100,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 100,
    }
}



// Change Order Of Arms --------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_order_of_arms(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        _ => 100,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_order_of_arms(x: u32) -> u32 {
    match x {
        1 => 1,
        0 => 0,
        _ => 100,
    }
}



// Add Guard Clause ------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1      => 1,
        _ => 100,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1 if y => 1,
        _ => 100,
    }
}



// Change Guard Clause ------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1 if  y => 1,
        _ => 100,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1 if !y => 1,
        _ => 100,
    }
}



// Add @-Binding ---------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
            _ => x,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        x @ _ => x,
    }
}



// Change Name of @-Binding ----------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_name_of_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        x @ _ => 7,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_name_of_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        y @ _ => 7,
    }
}



// Change Simple Binding To Pattern --------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_simple_name_to_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (0, 0) => 0,
         a     => 1,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_simple_name_to_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (0, 0) => 0,
        (x, y) => 1,
    }
}



// Change Name In Pattern ------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_name_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (a, 0) => 0,
        (a, 1) => a,
        _ => 100,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_name_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (b, 0) => 0,
        (a, 1) => a,
        _ => 100,
    }
}



// Change Mutability Of Binding In Pattern -------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_mutability_of_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (    a, 0) => 0,
        _ => 1,
    }
}

// Ignore optimized_mir in bpass2, the only change to optimized MIR is a span.
#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_mutability_of_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (mut a, 0) => 0,
        _ => 1,
    }
}



// Add `ref` To Binding In Pattern -------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_ref_to_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (    a, 0) => 0,
        _ => 1,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_ref_to_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (ref a, 0) => 0,
        _ => 1,
    }
}



// Add `&` To Binding In Pattern -------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_amp_to_binding_in_pattern(x: u32) -> u32 {
    match (&x, x & 1) {
        ( a, 0) => 0,
        _ => 1,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_amp_to_binding_in_pattern(x: u32) -> u32 {
    match (&x, x & 1) {
        (&a, 0) => 0,
        _ => 1,
    }
}



// Change RHS Of Arm -----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn change_rhs_of_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_rhs_of_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 3,
        _ => 2,
    }
}



// Add Alternative To Arm ------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
pub fn add_alternative_to_arm(x: u32) -> u32 {
    match x {
        0     => 0,
        1 => 1,
        _ => 2,
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_alternative_to_arm(x: u32) -> u32 {
    match x {
        0 | 7 => 0,
        1 => 3,
        _ => 2,
    }
}
