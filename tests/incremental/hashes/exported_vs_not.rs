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

// Case 1: The function body is not exported to metadata. If the body changes,
//         the hash of the owner node should change, but not the hash of
//         either the hir_owner or the Metadata node.

#[cfg(any(bpass1,bpass4))]
pub fn body_not_exported_to_metadata() -> u32 {
    1
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
pub fn body_not_exported_to_metadata() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         marked as #[inline]. Only the hash of the hir_owner depnode should be
//         unaffected by a change to the body.

#[cfg(any(bpass1,bpass4))]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    1
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         generic. Only the hash of the hir_owner depnode should be
//         unaffected by a change to the body.

#[cfg(any(bpass1,bpass4))]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    1
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,optimized_mir")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,optimized_mir")]
#[rustc_clean(cfg="bpass6")]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    2
}
