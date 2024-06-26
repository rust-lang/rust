// This test case tests the incremental compilation hash (ICH) implementation
// for `for` loops.

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


// Change loop body ------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_loop_body() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        break;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_loop_body() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 2;
        break;
    }
}



// Change iteration variable name ----------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_iteration_variable_name() {
    let mut _x = 0;
    for _i in 0..1 {
        _x = 1;
        break;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_iteration_variable_name() {
    let mut _x = 0;
    for _a in 0..1 {
        _x = 1;
        break;
    }
}



// Change iteration variable pattern -------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_iteration_variable_pattern() {
    let mut _x = 0;
    for  _i in &[0, 1, 2] {
        _x = 1;
        break;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir, typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir, typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_iteration_variable_pattern() {
    let mut _x = 0;
    for &_i in &[0, 1, 2] {
        _x = 1;
        break;
    }
}



// Change iterable -------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_iterable() {
    let mut _x = 0;
    for _ in &[0, 1, 2] {
        _x = 1;
        break;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, promoted_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, promoted_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_iterable() {
    let mut _x = 0;
    for _ in &[0, 1, 3] {
        _x = 1;
        break;
    }
}



// Add break -------------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        // ---
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir, typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir, typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        break;
    }
}



// Add loop label --------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_loop_label() {
    let mut _x = 0;
            for _ in 0..1 {
        _x = 1;
        break;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn add_loop_label() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        break;
    }
}



// Add loop label to break -----------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn add_loop_label_to_break() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        break       ;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn add_loop_label_to_break() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        break 'label;
    }
}



// Change break label ----------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_break_label() {
    let mut _x = 0;
    'outer: for _ in 0..1 {
        'inner: for _ in 0..1 {
            _x = 1;
            break 'inner;
        }
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
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
#[cfg(any(cfail1,cfail4))]
pub fn add_loop_label_to_continue() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        continue       ;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
pub fn add_loop_label_to_continue() {
    let mut _x = 0;
    'label: for _ in 0..1 {
        _x = 1;
        continue 'label;
    }
}



// Change continue label ----------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn change_continue_label() {
    let mut _x = 0;
    'outer: for _ in 0..1 {
        'inner: for _ in 0..1 {
            _x = 1;
            continue 'inner;
        }
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
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
#[cfg(any(cfail1,cfail4))]
pub fn change_continue_to_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        continue;
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_continue_to_break() {
    let mut _x = 0;
    for _ in 0..1 {
        _x = 1;
        break   ;
    }
}
