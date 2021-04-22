// This test case tests the incremental compilation hash (ICH) implementation
// for consts.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Change const visibility
#[cfg(cfail1)]
const CONST_VISIBILITY: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
pub const CONST_VISIBILITY: u8 = 0;


// Change type from i32 to u32
#[cfg(cfail1)]
const CONST_CHANGE_TYPE_1: i32 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_TYPE_1: u32 = 0;


// Change type from Option<u32> to Option<u64>
#[cfg(cfail1)]
const CONST_CHANGE_TYPE_2: Option<u32> = None;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_TYPE_2: Option<u64> = None;


// Change value between simple literals
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_1: i16 = {
    #[cfg(cfail1)]
    { 1 }

    #[cfg(not(cfail1))]
    { 2 }
};


// Change value between expressions
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_2: i16 = {
    #[cfg(cfail1)]
    { 1 + 1 }

    #[cfg(not(cfail1))]
    { 1 + 2 }
};

#[rustc_clean(cfg="cfail2", except="hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_3: i16 = {
    #[cfg(cfail1)]
    { 2 + 3 }

    #[cfg(not(cfail1))]
    { 2 * 3 }
};

#[rustc_clean(cfg="cfail2", except="hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_4: i16 = {
    #[cfg(cfail1)]
    { 1 + 2 * 3 }

    #[cfg(not(cfail1))]
    { 1 + 2 * 4 }
};


// Change type indirectly
struct ReferencedType1;
struct ReferencedType2;

mod const_change_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as Type;

    #[cfg(not(cfail1))]
    use super::ReferencedType2 as Type;

    #[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes,type_of")]
    #[rustc_clean(cfg="cfail3")]
    const CONST_CHANGE_TYPE_INDIRECTLY_1: Type = Type;

    #[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes,type_of")]
    #[rustc_clean(cfg="cfail3")]
    const CONST_CHANGE_TYPE_INDIRECTLY_2: Option<Type> = None;
}
