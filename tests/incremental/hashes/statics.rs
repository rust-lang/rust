// This test case tests the incremental compilation hash (ICH) implementation
// for statics.

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
#![feature(linkage)]
#![feature(thread_local)]
#![crate_type="rlib"]


// Change static visibility
#[cfg(any(bpass1,bpass4))]
static     STATIC_VISIBILITY: u8 = 0;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
pub static STATIC_VISIBILITY: u8 = 0;


// Change static mutability
#[cfg(any(bpass1,bpass4))]
static STATIC_MUTABILITY: u8 = 0;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
static mut STATIC_MUTABILITY: u8 = 0;


// Add linkage attribute
#[cfg(any(bpass1,bpass4))]
static STATIC_LINKAGE: u8 = 0;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
#[linkage="weak_odr"]
static STATIC_LINKAGE: u8 = 0;


// Add no_mangle attribute
#[cfg(any(bpass1,bpass4))]
static STATIC_NO_MANGLE: u8 = 0;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
#[unsafe(no_mangle)]
static STATIC_NO_MANGLE: u8 = 0;


// Add thread_local attribute
#[cfg(any(bpass1,bpass4))]
static STATIC_THREAD_LOCAL: u8 = 0;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
#[thread_local]
static STATIC_THREAD_LOCAL: u8 = 0;


// Change type from i16 to u64
#[cfg(any(bpass1,bpass4))]
static STATIC_CHANGE_TYPE_1: i16 = 0;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,type_of")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,type_of")]
#[rustc_clean(cfg="bpass6")]
static STATIC_CHANGE_TYPE_1: u64 = 0;


// Change type from Option<i8> to Option<u16>
#[cfg(any(bpass1,bpass4))]
static STATIC_CHANGE_TYPE_2: Option<i8> = None;

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,type_of")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,type_of")]
#[rustc_clean(cfg="bpass6")]
static STATIC_CHANGE_TYPE_2: Option<u16> = None;


// Change value between simple literals
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
static STATIC_CHANGE_VALUE_1: i16 = {
    #[cfg(any(bpass1,bpass4))]
    { 1 }

    #[cfg(not(any(bpass1,bpass4)))]
    { 2 }
};


// Change value between expressions
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
static STATIC_CHANGE_VALUE_2: i16 = {
    #[cfg(any(bpass1,bpass4))]
    { 1 + 1 }

    #[cfg(not(any(bpass1,bpass4)))]
    { 1 + 2 }
};

#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
static STATIC_CHANGE_VALUE_3: i16 = {
    #[cfg(any(bpass1,bpass4))]
    { 2 + 3 }

    #[cfg(not(any(bpass1,bpass4)))]
    { 2 * 3 }
};

#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
static STATIC_CHANGE_VALUE_4: i16 = {
    #[cfg(any(bpass1,bpass4))]
    { 1 + 2 * 3 }

    #[cfg(not(any(bpass1,bpass4)))]
    { 1 + 2 * 4 }
};


// Change type indirectly
struct ReferencedType1;
struct ReferencedType2;

mod static_change_type_indirectly {
    #[cfg(any(bpass1,bpass4))]
    use super::ReferencedType1 as Type;

    #[cfg(not(any(bpass1,bpass4)))]
    use super::ReferencedType2 as Type;

    #[rustc_clean(cfg="bpass2", except="owner,type_of")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,type_of")]
    #[rustc_clean(cfg="bpass6")]
    static STATIC_CHANGE_TYPE_INDIRECTLY_1: Type = Type;

    #[rustc_clean(cfg="bpass2", except="owner,type_of")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,type_of")]
    #[rustc_clean(cfg="bpass6")]
    static STATIC_CHANGE_TYPE_INDIRECTLY_2: Option<Type> = None;
}
