// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Case 1: The function body is not exported to metadata. If the body changes,
//         the hash of the HirBody node should change, but not the hash of
//         either the Hir or the Metadata node.

#[cfg(cfail1)]
pub fn body_not_exported_to_metadata() -> u32 {
    1
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,MirValidated,MirOptimized")]
#[rustc_clean(cfg="cfail3")]
pub fn body_not_exported_to_metadata() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         marked as #[inline]. Only the hash of the Hir depnode should be
//         unaffected by a change to the body.

#[cfg(cfail1)]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    1
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,MirValidated,MirOptimized")]
#[rustc_clean(cfg="cfail3")]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         generic. Only the hash of the Hir depnode should be
//         unaffected by a change to the body.

#[cfg(cfail1)]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    1
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,MirValidated,MirOptimized")]
#[rustc_clean(cfg="cfail3")]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    2
}

