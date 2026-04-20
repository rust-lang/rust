//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: bfail1 bfail2 bfail3 bfail4 bfail5 bfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [bfail1]compile-flags: -Zincremental-ignore-spans
//@ [bfail2]compile-flags: -Zincremental-ignore-spans
//@ [bfail3]compile-flags: -Zincremental-ignore-spans
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Case 1: The function body is not exported to metadata. If the body changes,
//         the hash of the opt_hir_owner_nodes node should change, but not the hash of
//         either the hir_owner or the Metadata node.

#[cfg(any(bfail1,bfail4))]
pub fn body_not_exported_to_metadata() -> u32 {
    1
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bfail6")]
pub fn body_not_exported_to_metadata() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         marked as #[inline]. Only the hash of the hir_owner depnode should be
//         unaffected by a change to the body.

#[cfg(any(bfail1,bfail4))]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    1
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bfail6")]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         generic. Only the hash of the hir_owner depnode should be
//         unaffected by a change to the body.

#[cfg(any(bfail1,bfail4))]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    1
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="bfail6")]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    2
}
