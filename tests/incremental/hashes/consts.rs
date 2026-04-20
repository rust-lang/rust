// This test case tests the incremental compilation hash (ICH) implementation
// for consts.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: bfail1 bfail2 bfail3
//@ compile-flags: -Z query-dep-graph -O
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Change const visibility
#[cfg(bfail1)]
const CONST_VISIBILITY: u8 = 0;

#[cfg(not(bfail1))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
pub const CONST_VISIBILITY: u8 = 0;


// Change type from i32 to u32
#[cfg(bfail1)]
const CONST_CHANGE_TYPE_1: i32 = 0;

#[cfg(not(bfail1))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="bfail3")]
const CONST_CHANGE_TYPE_1: u32 = 0;


// Change type from Option<u32> to Option<u64>
#[cfg(bfail1)]
const CONST_CHANGE_TYPE_2: Option<u32> = None;

#[cfg(not(bfail1))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="bfail3")]
const CONST_CHANGE_TYPE_2: Option<u64> = None;


// Change value between simple literals
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
const CONST_CHANGE_VALUE_1: i16 = {
    #[cfg(bfail1)]
    { 1 }

    #[cfg(not(bfail1))]
    { 2 }
};


// Change value between expressions
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
const CONST_CHANGE_VALUE_2: i16 = {
    #[cfg(bfail1)]
    { 1 + 1 }

    #[cfg(not(bfail1))]
    { 1 + 2 }
};

#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
const CONST_CHANGE_VALUE_3: i16 = {
    #[cfg(bfail1)]
    { 2 + 3 }

    #[cfg(not(bfail1))]
    { 2 * 3 }
};

#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="bfail3")]
const CONST_CHANGE_VALUE_4: i16 = {
    #[cfg(bfail1)]
    { 1 + 2 * 3 }

    #[cfg(not(bfail1))]
    { 1 + 2 * 4 }
};


// Change type indirectly
struct ReferencedType1;
struct ReferencedType2;

mod const_change_type_indirectly {
    #[cfg(bfail1)]
    use super::ReferencedType1 as Type;

    #[cfg(not(bfail1))]
    use super::ReferencedType2 as Type;

    #[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,type_of")]
    #[rustc_clean(cfg="bfail3")]
    const CONST_CHANGE_TYPE_INDIRECTLY_1: Type = Type;

    #[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,type_of")]
    #[rustc_clean(cfg="bfail3")]
    const CONST_CHANGE_TYPE_INDIRECTLY_2: Option<Type> = None;
}
