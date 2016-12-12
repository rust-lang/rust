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
// for exprs that can panic at runtime (e.g. because of bounds checking). For
// these expressions an error message containing their source location is
// generated, so their hash must always depend on their location in the source
// code, not just when debuginfo is enabled.

// As opposed to the panic_exprs.rs test case, this test case checks that things
// behave as expected when overflow checks are off:
//
// - Addition, subtraction, and multiplication do not change the ICH, unless
//   the function containing them is marked with rustc_inherit_overflow_checks.
// - Division by zero and bounds checks always influence the ICH

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Z force-overflow-checks=off

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Indexing expression ---------------------------------------------------------
#[cfg(cfail1)]
pub fn indexing(slice: &[u8]) -> u8 {
    slice[100]
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn indexing(slice: &[u8]) -> u8 {
    slice[100]
}


// Arithmetic overflow plus ----------------------------------------------------
#[cfg(cfail1)]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_plus_inherit(val: i32) -> i32 {
    val + 1
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_plus_inherit(val: i32) -> i32 {
    val + 1
}


// Arithmetic overflow minus ----------------------------------------------------
#[cfg(cfail1)]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_minus_inherit(val: i32) -> i32 {
    val - 1
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_minus_inherit(val: i32) -> i32 {
    val - 1
}


// Arithmetic overflow mult ----------------------------------------------------
#[cfg(cfail1)]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_mult_inherit(val: i32) -> i32 {
    val * 2
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_mult_inherit(val: i32) -> i32 {
    val * 2
}


// Arithmetic overflow negation ------------------------------------------------
#[cfg(cfail1)]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_negation_inherit(val: i32) -> i32 {
    -val
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[rustc_inherit_overflow_checks]
pub fn arithmetic_overflow_negation_inherit(val: i32) -> i32 {
    -val
}


// Division by zero ------------------------------------------------------------
#[cfg(cfail1)]
pub fn division_by_zero(val: i32) -> i32 {
    2 / val
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn division_by_zero(val: i32) -> i32 {
    2 / val
}

// Division by zero ------------------------------------------------------------
#[cfg(cfail1)]
pub fn mod_by_zero(val: i32) -> i32 {
    2 % val
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn mod_by_zero(val: i32) -> i32 {
    2 % val
}



// THE FOLLOWING ITEMS SHOULD NOT BE INFLUENCED BY THEIR SOURCE LOCATION

// bitwise ---------------------------------------------------------------------
#[cfg(cfail1)]
pub fn bitwise(val: i32) -> i32 {
    !val & 0x101010101 | 0x45689 ^ 0x2372382 << 1 >> 1
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn bitwise(val: i32) -> i32 {
    !val & 0x101010101 | 0x45689 ^ 0x2372382 << 1 >> 1
}


// logical ---------------------------------------------------------------------
#[cfg(cfail1)]
pub fn logical(val1: bool, val2: bool, val3: bool) -> bool {
    val1 && val2 || val3
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn logical(val1: bool, val2: bool, val3: bool) -> bool {
    val1 && val2 || val3
}

// Arithmetic overflow plus ----------------------------------------------------
#[cfg(cfail1)]
pub fn arithmetic_overflow_plus(val: i32) -> i32 {
    val + 1
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn arithmetic_overflow_plus(val: i32) -> i32 {
    val + 1
}


// Arithmetic overflow minus ----------------------------------------------------
#[cfg(cfail1)]
pub fn arithmetic_overflow_minus(val: i32) -> i32 {
    val - 1
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn arithmetic_overflow_minus(val: i32) -> i32 {
    val - 1
}


// Arithmetic overflow mult ----------------------------------------------------
#[cfg(cfail1)]
pub fn arithmetic_overflow_mult(val: i32) -> i32 {
    val * 2
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn arithmetic_overflow_mult(val: i32) -> i32 {
    val * 2
}


// Arithmetic overflow negation ------------------------------------------------
#[cfg(cfail1)]
pub fn arithmetic_overflow_negation(val: i32) -> i32 {
    -val
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn arithmetic_overflow_negation(val: i32) -> i32 {
    -val
}
