// This test case tests the incremental compilation hash (ICH) implementation
// for indexing expression.

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

// Change simple index
#[cfg(any(cfail1,cfail4))]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[3]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[4]
}



// Change lower bound
#[cfg(any(cfail1,cfail4))]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[2..5]
}



// Change upper bound
#[cfg(any(cfail1,cfail4))]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Add lower bound
#[cfg(any(cfail1,cfail4))]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[ ..4]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..4]
}



// Add upper bound
#[cfg(any(cfail1,cfail4))]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3.. ]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Change mutability
#[cfg(any(cfail1,cfail4))]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&mut slice[3..5])[0]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&    slice[3..5])[0]
}



// Exclusive to inclusive range
#[cfg(any(cfail1,cfail4))]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3.. 7]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3..=7]
}
