//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [cfail1]compile-flags: -Zincremental-ignore-spans
//@ [cfail2]compile-flags: -Zincremental-ignore-spans
//@ [cfail3]compile-flags: -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Case 1: The function body is not exported to metadata. If the body changes,
//         the hash of the opt_hir_owner_nodes node should change, but not the hash of
//         either the hir_owner or the Metadata node.

#[cfg(any(cfail1,cfail4))]
pub fn body_not_exported_to_metadata() -> u32 {
    1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn body_not_exported_to_metadata() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         marked as #[inline]. Only the hash of the hir_owner depnode should be
//         unaffected by a change to the body.

#[cfg(any(cfail1,cfail4))]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[inline]
pub fn body_exported_to_metadata_because_of_inline() -> u32 {
    2
}



// Case 2: The function body *is* exported to metadata because the function is
//         generic. Only the hash of the hir_owner depnode should be
//         unaffected by a change to the body.

#[cfg(any(cfail1,cfail4))]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[inline]
pub fn body_exported_to_metadata_because_of_generic() -> u32 {
    2
}
