// This test case tests the incremental compilation hash (ICH) implementation
// for `for` loops.

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


// Change loop body ------------------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_loop_body() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        break;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn change_loop_body() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 2;
        break;
    }
}



// Change iteration variable name ----------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_iteration_variable_name() {
    let mut _x = 0;
    for _i in 0..1 {
        _x = 1;
        break;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn change_iteration_variable_name() {
    let mut _x = 0;
    for _a in 0..1 {
        _x = 1;
        break;
    }
}



// Change iteration variable pattern -------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_iteration_variable_pattern() {
    let mut _x = 0;
    for  _i in &[0, 1, 2] {
        _x = 1;
        break;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir, typeck_root")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir, typeck_root")]
#[rustc_clean(cfg="bfail6")]
pub fn change_iteration_variable_pattern() {
    let mut _x = 0;
    for &_i in &[0, 1, 2] {
        _x = 1;
        break;
    }
}



// Change iterable -------------------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_iterable() {
    let mut _x = 0;
    for _ in &[0, 1, 2] {
        _x = 1;
        break;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, promoted_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, promoted_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn change_iterable() {
    let mut _x = 0;
    for _ in &[0, 1, 3] {
        _x = 1;
        break;
    }
}



// Add break -------------------------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn add_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        // ---
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir, typeck_root")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir, typeck_root")]
#[rustc_clean(cfg="bfail6")]
pub fn add_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        break;
    }
}



// Add loop label --------------------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn add_loop_label() {
    let mut _x = 0;
            for _ in 0..1 {
        _x = 1;
        break;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn add_loop_label() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        break;
    }
}



// Add loop label to break -----------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn add_loop_label_to_break() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        break       ;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn add_loop_label_to_break() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        break 'label;
    }
}



// Change break label ----------------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_break_label() {
    let mut _x = 0;
    'outer: for _ in 0..1 {
        'inner: for _ in 0..1 {
            _x = 1;
            break 'inner;
        }
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn change_break_label() {
    let mut _x = 0;
    'outer: for _ in 0..1 {
        'inner: for _ in 0..1 {
            _x = 1;
            break 'outer;
        }
    }
}



// Add loop label to continue --------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn add_loop_label_to_continue() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        continue       ;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail6")]
pub fn add_loop_label_to_continue() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        continue 'label;
    }
}



// Change continue label ----------------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_continue_label() {
    let mut _x = 0;
    'outer: for _ in 0..1 {
        'inner: for _ in 0..1 {
            _x = 1;
            continue 'inner;
        }
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn change_continue_label() {
    let mut _x = 0;
    'outer: for _ in 0..1 {
        'inner: for _ in 0..1 {
            _x = 1;
            continue 'outer;
        }
    }
}



// Change continue to break ----------------------------------------------------
#[cfg(any(bfail1,bfail4))]
pub fn change_continue_to_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        continue;
    }
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn change_continue_to_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        break   ;
    }
}
