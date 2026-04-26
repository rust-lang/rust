// This test case tests the incremental compilation hash (ICH) implementation
// for indexing expression.

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

// Change simple index
#[cfg(any(bpass1,bpass4))]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[3]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[4]
}



// Change lower bound
#[cfg(any(bpass1,bpass4))]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[2..5]
}



// Change upper bound
#[cfg(any(bpass1,bpass4))]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Add lower bound
#[cfg(any(bpass1,bpass4))]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[ ..4]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..4]
}



// Add upper bound
#[cfg(any(bpass1,bpass4))]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3.. ]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Change mutability
#[cfg(any(bpass1,bpass4))]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&mut slice[3..5])[0]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&    slice[3..5])[0]
}



// Exclusive to inclusive range
#[cfg(any(bpass1,bpass4))]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3.. 7]
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass2")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3..=7]
}
