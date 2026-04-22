// This test case tests the incremental compilation hash (ICH) implementation
// for closure expression.

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


// Change closure body
#[cfg(any(bpass1,bpass4))]
pub fn change_closure_body() {
    let _ = || 1u32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bpass6")]
pub fn change_closure_body() {
    let _ = || 3u32;
}



// Add parameter
#[cfg(any(bpass1,bpass4))]
pub fn add_parameter() {
    let x = 0u32;
    let _ = |      | x + 1;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes, typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes, typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_parameter() {
    let x = 0u32;
    let _ = |x: u32| x + 1;
}



// Change parameter pattern
#[cfg(any(bpass1,bpass4))]
pub fn change_parameter_pattern() {
    let _ = | x  : (u32,)| x;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes, typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes, typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_parameter_pattern() {
    let _ = |(x,): (u32,)| x;
}



// Add `move` to closure
#[cfg(any(bpass1,bpass4))]
pub fn add_move() {
    let _ =      || 1;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bpass6")]
pub fn add_move() {
    let _ = move || 1;
}



// Add type ascription to parameter
#[cfg(any(bpass1,bpass4))]
pub fn add_type_ascription_to_parameter() {
    let closure = |x     | x + 1u32;
    let _: u32 = closure(1);
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "opt_hir_owner_nodes, typeck_root")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "opt_hir_owner_nodes, typeck_root")]
#[rustc_clean(cfg = "bpass6")]
pub fn add_type_ascription_to_parameter() {
    let closure = |x: u32| x + 1u32;
    let _: u32 = closure(1);
}



// Change parameter type
#[cfg(any(bpass1,bpass4))]
pub fn change_parameter_type() {
    let closure = |x: u32| (x as u64) + 1;
    let _ = closure(1);
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes, optimized_mir, typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes, optimized_mir, typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_parameter_type() {
    let closure = |x: u16| (x as u64) + 1;
    let _ = closure(1);
}
