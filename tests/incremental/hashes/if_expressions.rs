// This test case tests the incremental compilation hash (ICH) implementation
// for if expressions.

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

// Change condition (if)
#[cfg(any(bpass1,bpass4))]
pub fn change_condition(x: bool) -> u32 {
    if  x {
        return 1
    }

    return 0
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_condition(x: bool) -> u32 {
    if !x {
        return 1
    }

    return 0
}

// Change then branch (if)
#[cfg(any(bpass1,bpass4))]
pub fn change_then_branch(x: bool) -> u32 {
    if x {
        return 1
    }

    return 0
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_then_branch(x: bool) -> u32 {
    if x {
        return 2
    }

    return 0
}



// Change else branch (if)
#[cfg(any(bpass1,bpass4))]
pub fn change_else_branch(x: bool) -> u32 {
    if x {
        1
    } else {
        2
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_else_branch(x: bool) -> u32 {
    if x {
        1
    } else {
        3
    }
}



// Add else branch (if)
#[cfg(any(bpass1,bpass4))]
pub fn add_else_branch(x: bool) -> u32 {
    let mut ret = 1;

    if x {
        ret = 2;
    /*----*/
    }

    ret
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn add_else_branch(x: bool) -> u32 {
    let mut ret = 1;

    if x {
        ret = 2;
    } else {
    }

    ret
}



// Change condition (if let)
#[cfg(any(bpass1,bpass4))]
pub fn change_condition_if_let(x: Option<u32>) -> u32 {
    if let Some(_x) = x {
        return 1
    }

    0
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_condition_if_let(x: Option<u32>) -> u32 {
    if let Some(_ ) = x {
        return 1
    }

    0
}



// Change then branch (if let)
#[cfg(any(bpass1,bpass4))]
pub fn change_then_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        return x //-
    }

    0
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir,typeck_root")]
#[rustc_clean(cfg="bpass6")]
pub fn change_then_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        return x + 1
    }

    0
}



// Change else branch (if let)
#[cfg(any(bpass1,bpass4))]
pub fn change_else_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        x
    } else {
        1
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn change_else_branch_if_let(x: Option<u32>) -> u32 {
    if let Some(x) = x {
        x
    } else {
        2
    }
}



// Add else branch (if let)
#[cfg(any(bpass1,bpass4))]
pub fn add_else_branch_if_let(x: Option<u32>) -> u32 {
    let mut ret = 1;

    if let Some(x) = x {
        ret = x;
    /*----*/
    }

    ret
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="opt_hir_owner_nodes,typeck_root")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="opt_hir_owner_nodes,typeck_root,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn add_else_branch_if_let(x: Option<u32>) -> u32 {
    let mut ret = 1;

    if let Some(x) = x {
        ret = x;
    } else {
    }

    ret
}
