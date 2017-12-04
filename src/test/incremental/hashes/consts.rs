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
// for consts.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Change const visibility ---------------------------------------------------
#[cfg(cfail1)]
const CONST_VISIBILITY: u8 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
pub const CONST_VISIBILITY: u8 = 0;


// Change type from i32 to u32 ------------------------------------------------
#[cfg(cfail1)]
const CONST_CHANGE_TYPE_1: i32 = 0;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,TypeOfItem")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_TYPE_1: u32 = 0;


// Change type from Option<u32> to Option<u64> --------------------------------
#[cfg(cfail1)]
const CONST_CHANGE_TYPE_2: Option<u32> = None;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,TypeOfItem")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_TYPE_2: Option<u64> = None;


// Change value between simple literals ---------------------------------------
#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_1: i16 = {
    #[cfg(cfail1)]
    { 1 }

    #[cfg(not(cfail1))]
    { 2 }
};


// Change value between expressions -------------------------------------------
#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_2: i16 = {
    #[cfg(cfail1)]
    { 1 + 1 }

    #[cfg(not(cfail1))]
    { 1 + 2 }
};

#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_3: i16 = {
    #[cfg(cfail1)]
    { 2 + 3 }

    #[cfg(not(cfail1))]
    { 2 * 3 }
};

#[rustc_clean(cfg="cfail2", except="HirBody")]
#[rustc_clean(cfg="cfail3")]
const CONST_CHANGE_VALUE_4: i16 = {
    #[cfg(cfail1)]
    { 1 + 2 * 3 }

    #[cfg(not(cfail1))]
    { 1 + 2 * 4 }
};


// Change type indirectly -----------------------------------------------------
struct ReferencedType1;
struct ReferencedType2;

mod const_change_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as Type;

    #[cfg(not(cfail1))]
    use super::ReferencedType2 as Type;

    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,TypeOfItem")]
    #[rustc_clean(cfg="cfail3")]
    const CONST_CHANGE_TYPE_INDIRECTLY_1: Type = Type;

    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,TypeOfItem")]
    #[rustc_clean(cfg="cfail3")]
    const CONST_CHANGE_TYPE_INDIRECTLY_2: Option<Type> = None;
}
