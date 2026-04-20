// This test case tests the incremental compilation hash (ICH) implementation
// for indexing expression.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: bfail1 bfail2 bfail3 bfail4 bfail5 bfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [bfail1]compile-flags: -Zincremental-ignore-spans
//@ [bfail2]compile-flags: -Zincremental-ignore-spans
//@ [bfail3]compile-flags: -Zincremental-ignore-spans
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Change simple index
#[cfg(any(bfail1,bfail4))]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[3]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[4]
}



// Change lower bound
#[cfg(any(bfail1,bfail4))]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[2..5]
}



// Change upper bound
#[cfg(any(bfail1,bfail4))]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Add lower bound
#[cfg(any(bfail1,bfail4))]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[ ..4]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..4]
}



// Add upper bound
#[cfg(any(bfail1,bfail4))]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3.. ]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Change mutability
#[cfg(any(bfail1,bfail4))]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&mut slice[3..5])[0]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&    slice[3..5])[0]
}



// Exclusive to inclusive range
#[cfg(any(bfail1,bfail4))]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3.. 7]
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck_root", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3..=7]
}
