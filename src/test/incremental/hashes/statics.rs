// This test case tests the incremental compilation hash (ICH) implementation
// for statics.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(linkage)]
#![feature(thread_local)]
#![crate_type="rlib"]


// Change static visibility ---------------------------------------------------
#[cfg(cfail1)]
static STATIC_VISIBILITY: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
pub static STATIC_VISIBILITY: u8 = 0;


// Change static mutability ---------------------------------------------------
#[cfg(cfail1)]
static STATIC_MUTABILITY: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
static mut STATIC_MUTABILITY: u8 = 0;


// Add linkage attribute ------------------------------------------------------
#[cfg(cfail1)]
static STATIC_LINKAGE: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
#[linkage="weak_odr"]
static STATIC_LINKAGE: u8 = 0;


// Add no_mangle attribute ----------------------------------------------------
#[cfg(cfail1)]
static STATIC_NO_MANGLE: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
#[no_mangle]
static STATIC_NO_MANGLE: u8 = 0;


// Add thread_local attribute -------------------------------------------------
#[cfg(cfail1)]
static STATIC_THREAD_LOCAL: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
#[thread_local]
static STATIC_THREAD_LOCAL: u8 = 0;


// Change type from i16 to u64 ------------------------------------------------
#[cfg(cfail1)]
static STATIC_CHANGE_TYPE_1: i16 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,type_of")]
#[rustc_clean(cfg="cfail3")]
static STATIC_CHANGE_TYPE_1: u64 = 0;


// Change type from Option<i8> to Option<u16> ---------------------------------
#[cfg(cfail1)]
static STATIC_CHANGE_TYPE_2: Option<i8> = None;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,type_of")]
#[rustc_clean(cfg="cfail3")]
static STATIC_CHANGE_TYPE_2: Option<u16> = None;


// Change value between simple literals ---------------------------------------
#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
static STATIC_CHANGE_VALUE_1: i16 = {
    #[cfg(cfail1)]
    { 1 }

    #[cfg(not(cfail1))]
    { 2 }
};


// Change value between expressions -------------------------------------------
#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
static STATIC_CHANGE_VALUE_2: i16 = {
    #[cfg(cfail1)]
    { 1 + 1 }

    #[cfg(not(cfail1))]
    { 1 + 2 }
};

#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
static STATIC_CHANGE_VALUE_3: i16 = {
    #[cfg(cfail1)]
    { 2 + 3 }

    #[cfg(not(cfail1))]
    { 2 * 3 }
};

#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
static STATIC_CHANGE_VALUE_4: i16 = {
    #[cfg(cfail1)]
    { 1 + 2 * 3 }

    #[cfg(not(cfail1))]
    { 1 + 2 * 4 }
};


// Change type indirectly -----------------------------------------------------
struct ReferencedType1;
struct ReferencedType2;

mod static_change_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as Type;

    #[cfg(not(cfail1))]
    use super::ReferencedType2 as Type;

    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,type_of")]
    #[rustc_clean(cfg="cfail3")]
    static STATIC_CHANGE_TYPE_INDIRECTLY_1: Type = Type;

    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,type_of")]
    #[rustc_clean(cfg="cfail3")]
    static STATIC_CHANGE_TYPE_INDIRECTLY_2: Option<Type> = None;
}
