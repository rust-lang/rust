// This test case tests the incremental compilation hash (ICH) implementation
// for closure expression.

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


// Change closure body
#[cfg(any(cfail1,cfail4))]
pub fn change_closure_body() {
    let _ = || 1u32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
pub fn change_closure_body() {
    let _ = || 3u32;
}



// Add parameter
#[cfg(any(cfail1,cfail4))]
pub fn add_parameter() {
    let x = 0u32;
    let _ = |      | x + 1;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_parameter() {
    let x = 0u32;
    let _ = |x: u32| x + 1;
}



// Change parameter pattern
#[cfg(any(cfail1,cfail4))]
pub fn change_parameter_pattern() {
    let _ = | x  : (u32,)| x;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_parameter_pattern() {
    let _ = |(x,): (u32,)| x;
}



// Add `move` to closure
#[cfg(any(cfail1,cfail4))]
pub fn add_move() {
    let _ =      || 1;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
pub fn add_move() {
    let _ = move || 1;
}



// Add type ascription to parameter
#[cfg(any(cfail1,cfail4))]
pub fn add_type_ascription_to_parameter() {
    let closure = |x     | x + 1u32;
    let _: u32 = closure(1);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2", except = "opt_hir_owner_nodes, typeck")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5", except = "opt_hir_owner_nodes, typeck")]
#[rustc_clean(cfg = "cfail6")]
pub fn add_type_ascription_to_parameter() {
    let closure = |x: u32| x + 1u32;
    let _: u32 = closure(1);
}



// Change parameter type
#[cfg(any(cfail1,cfail4))]
pub fn change_parameter_type() {
    let closure = |x: u32| (x as u64) + 1;
    let _ = closure(1);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes, optimized_mir, typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes, optimized_mir, typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_parameter_type() {
    let closure = |x: u16| (x as u64) + 1;
    let _ = closure(1);
}
