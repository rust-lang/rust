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
// for if expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph


#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Change condition (if) -------------------------------------------------------
#[cfg(cfail1)]
pub fn change_condition(x: bool) -> u32 {
    if x {
        return 1
    }

    return 0
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn change_condition(x: bool) -> u32 {
    if !x {
        return 1
    }

    return 0
}

// Change then branch (if) -----------------------------------------------------
#[cfg(cfail1)]
pub fn change_then_branch(x: bool) -> u32 {
    if x {
        return 1
    }

    return 0
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn change_then_branch(x: bool) -> u32 {
    if x {
        return 2
    }

    return 0
}



// Change else branch (if) -----------------------------------------------------
#[cfg(cfail1)]
pub fn change_else_branch(x: bool) -> u32 {
    if x {
        1
    } else {
        2
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn change_else_branch(x: bool) -> u32 {
    if x {
        1
    } else {
        3
    }
}



// Add else branch (if) --------------------------------------------------------
#[cfg(cfail1)]
pub fn add_else_branch(x: bool) -> u32 {
    let mut ret = 1;

    if x {
        ret += 1;
    }

    ret
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn add_else_branch(x: bool) -> u32 {
    let mut ret = 1;

    if x {
        ret += 1;
    } else {
    }

    ret
}



// Change condition (if let) ---------------------------------------------------
#[cfg(cfail1)]
pub fn change_condition_if_let(x: Option<u32>) -> u32 {
    if let Some(_x) = x {
        return 1
    }

    0
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn change_condition_if_let(x: Option<u32>) -> u32 {
    if let Some(_) = x {
        return 1
    }

    0
}



// Change then branch (if let) -------------------------------------------------
#[cfg(cfail1)]
pub fn change_then_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        return x
    }

    0
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn change_then_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        return x + 1
    }

    0
}



// Change else branch (if let) -------------------------------------------------
#[cfg(cfail1)]
pub fn change_else_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        x
    } else {
        1
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn change_else_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        x
    } else {
        2
    }
}



// Add else branch (if let) ----------------------------------------------------
#[cfg(cfail1)]
pub fn add_else_branch_if_let(x: Option<u32>) -> u32 {
    let mut ret = 1;

    if let Some(x) = x {
        ret += x;
    }

    ret
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub fn add_else_branch_if_let(x: Option<u32>) -> u32 {
    let mut ret = 1;

    if let Some(x) = x {
        ret += x;
    } else {
    }

    ret
}
