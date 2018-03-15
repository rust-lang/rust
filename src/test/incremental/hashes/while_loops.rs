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
// for `while` loops.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Change loop body ------------------------------------------------------------
#[cfg(cfail1)]
fn change_loop_body() {
    let mut _x = 0;
    while true {
        _x = 1;
        break;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_loop_body() {
    let mut _x = 0;
    while true {
        _x = 2;
        break;
    }
}



// Change loop body ------------------------------------------------------------
#[cfg(cfail1)]
fn change_loop_condition() {
    let mut _x = 0;
    while true {
        _x = 1;
        break;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_loop_condition() {
    let mut _x = 0;
    while false {
        _x = 1;
        break;
    }
}



// Add break -------------------------------------------------------------------
#[cfg(cfail1)]
fn add_break() {
    let mut _x = 0;
    while true {
        _x = 1;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_break() {
    let mut _x = 0;
    while true {
        _x = 1;
        break;
    }
}



// Add loop label --------------------------------------------------------------
#[cfg(cfail1)]
fn add_loop_label() {
    let mut _x = 0;
    while true {
        _x = 1;
        break;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_loop_label() {
    let mut _x = 0;
    'label: while true {
        _x = 1;
        break;
    }
}



// Add loop label to break -----------------------------------------------------
#[cfg(cfail1)]
fn add_loop_label_to_break() {
    let mut _x = 0;
    'label: while true {
        _x = 1;
        break;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_loop_label_to_break() {
    let mut _x = 0;
    'label: while true {
        _x = 1;
        break 'label;
    }
}



// Change break label ----------------------------------------------------------
#[cfg(cfail1)]
fn change_break_label() {
    let mut _x = 0;
    'outer: while true {
        'inner: while true {
            _x = 1;
            break 'inner;
        }
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_break_label() {
    let mut _x = 0;
    'outer: while true {
        'inner: while true {
            _x = 1;
            break 'outer;
        }
    }
}



// Add loop label to continue --------------------------------------------------
#[cfg(cfail1)]
fn add_loop_label_to_continue() {
    let mut _x = 0;
    'label: while true {
        _x = 1;
        continue;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_loop_label_to_continue() {
    let mut _x = 0;
    'label: while true {
        _x = 1;
        continue 'label;
    }
}



// Change continue label ----------------------------------------------------------
#[cfg(cfail1)]
fn change_continue_label() {
    let mut _x = 0;
    'outer: while true {
        'inner: while true {
            _x = 1;
            continue 'inner;
        }
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_continue_label() {
    let mut _x = 0;
    'outer: while true {
        'inner: while true {
            _x = 1;
            continue 'outer;
        }
    }
}



// Change continue to break ----------------------------------------------------
#[cfg(cfail1)]
fn change_continue_to_break() {
    let mut _x = 0;
    while true {
        _x = 1;
        continue;
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_continue_to_break() {
    let mut _x = 0;
    while true {
        _x = 1;
        break;
    }
}
