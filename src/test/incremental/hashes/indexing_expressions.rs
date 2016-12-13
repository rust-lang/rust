// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test case tests the incremental compilation hash (ICH) implementation
// for closure expression.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]
#![feature(inclusive_range_syntax)]

// Change simple index ---------------------------------------------------------
#[cfg(cfail1)]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[3]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_simple_index(slice: &[u32]) -> u32 {
    slice[4]
}



// Change lower bound ----------------------------------------------------------
#[cfg(cfail1)]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[2..5]
}



// Change upper bound ----------------------------------------------------------
#[cfg(cfail1)]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..5]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Add lower bound -------------------------------------------------------------
#[cfg(cfail1)]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[..4]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_lower_bound(slice: &[u32]) -> &[u32] {
    &slice[3..4]
}



// Add upper bound -------------------------------------------------------------
#[cfg(cfail1)]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_upper_bound(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}



// Change mutability -----------------------------------------------------------
#[cfg(cfail1)]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&mut slice[3..5])[0]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_mutability(slice: &mut [u32]) -> u32 {
    (&slice[3..5])[0]
}



// Exclusive to inclusive range ------------------------------------------------
#[cfg(cfail1)]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3..7]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn exclusive_to_inclusive_range(slice: &[u32]) -> &[u32] {
    &slice[3...7]
}
