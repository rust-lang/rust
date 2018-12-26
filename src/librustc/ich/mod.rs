//! ICH - Incremental Compilation Hash

crate use rustc_data_structures::fingerprint::Fingerprint;
pub use self::caching_source_map_view::CachingSourceMapView;
pub use self::hcx::{StableHashingContextProvider, StableHashingContext, NodeIdHashingMode,
                    hash_stable_trait_impls};
mod caching_source_map_view;
mod hcx;

mod impls_cstore;
mod impls_hir;
mod impls_mir;
mod impls_misc;
mod impls_ty;
mod impls_syntax;

pub const ATTR_DIRTY: &str = "rustc_dirty";
pub const ATTR_CLEAN: &str = "rustc_clean";
pub const ATTR_IF_THIS_CHANGED: &str = "rustc_if_this_changed";
pub const ATTR_THEN_THIS_WOULD_NEED: &str = "rustc_then_this_would_need";
pub const ATTR_PARTITION_REUSED: &str = "rustc_partition_reused";
pub const ATTR_PARTITION_CODEGENED: &str = "rustc_partition_codegened";
pub const ATTR_EXPECTED_CGU_REUSE: &str = "rustc_expected_cgu_reuse";

pub const IGNORED_ATTRIBUTES: &[&str] = &[
    "cfg",
    ATTR_IF_THIS_CHANGED,
    ATTR_THEN_THIS_WOULD_NEED,
    ATTR_DIRTY,
    ATTR_CLEAN,
    ATTR_PARTITION_REUSED,
    ATTR_PARTITION_CODEGENED,
    ATTR_EXPECTED_CGU_REUSE,
];
