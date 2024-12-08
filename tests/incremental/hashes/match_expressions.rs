// This test case tests the incremental compilation hash (ICH) implementation
// for match expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [cfail1]compile-flags: -Zincremental-ignore-spans
//@ [cfail2]compile-flags: -Zincremental-ignore-spans
//@ [cfail3]compile-flags: -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Add Arm ---------------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        /*---*/
        _ => 100,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 100,
    }
}



// Change Order Of Arms --------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_order_of_arms(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        _ => 100,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_order_of_arms(x: u32) -> u32 {
    match x {
        1 => 1,
        0 => 0,
        _ => 100,
    }
}



// Add Guard Clause ------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1      => 1,
        _ => 100,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1 if y => 1,
        _ => 100,
    }
}



// Change Guard Clause ------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1 if  y => 1,
        _ => 100,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_guard_clause(x: u32, y: bool) -> u32 {
    match x {
        0 => 0,
        1 if !y => 1,
        _ => 100,
    }
}



// Add @-Binding ---------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
            _ => x,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        x @ _ => x,
    }
}



// Change Name of @-Binding ----------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_name_of_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        x @ _ => 7,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_name_of_at_binding(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        y @ _ => 7,
    }
}



// Change Simple Binding To Pattern --------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_simple_name_to_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (0, 0) => 0,
         a     => 1,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_simple_name_to_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (0, 0) => 0,
        (x, y) => 1,
    }
}



// Change Name In Pattern ------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_name_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (a, 0) => 0,
        (a, 1) => a,
        _ => 100,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_name_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (b, 0) => 0,
        (a, 1) => a,
        _ => 100,
    }
}



// Change Mutability Of Binding In Pattern -------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_mutability_of_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (    a, 0) => 0,
        _ => 1,
    }
}

// Ignore optimized_mir in cfail2, the only change to optimized MIR is a span.
#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_mutability_of_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (mut a, 0) => 0,
        _ => 1,
    }
}



// Add `ref` To Binding In Pattern -------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_ref_to_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (    a, 0) => 0,
        _ => 1,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_ref_to_binding_in_pattern(x: u32) -> u32 {
    match (x, x & 1) {
        (ref a, 0) => 0,
        _ => 1,
    }
}



// Add `&` To Binding In Pattern -------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_amp_to_binding_in_pattern(x: u32) -> u32 {
    match (&x, x & 1) {
        ( a, 0) => 0,
        _ => 1,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_amp_to_binding_in_pattern(x: u32) -> u32 {
    match (&x, x & 1) {
        (&a, 0) => 0,
        _ => 1,
    }
}



// Change RHS Of Arm -----------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_rhs_of_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_rhs_of_arm(x: u32) -> u32 {
    match x {
        0 => 0,
        1 => 3,
        _ => 2,
    }
}



// Add Alternative To Arm ------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_alternative_to_arm(x: u32) -> u32 {
    match x {
        0     => 0,
        1 => 1,
        _ => 2,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_alternative_to_arm(x: u32) -> u32 {
    match x {
        0 | 7 => 0,
        1 => 3,
        _ => 2,
    }
}
