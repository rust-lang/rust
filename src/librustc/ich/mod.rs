//! ICH - Incremental Compilation Hash

crate use rustc_data_structures::fingerprint::Fingerprint;
pub use self::caching_source_map_view::CachingSourceMapView;
pub use self::hcx::{StableHashingContextProvider, StableHashingContext, NodeIdHashingMode,
                    hash_stable_trait_impls};
use syntax::symbol::{Symbol, sym};

mod caching_source_map_view;
mod hcx;

mod impls_hir;
mod impls_misc;
mod impls_ty;
mod impls_syntax;

pub const ATTR_DIRTY: Symbol = sym::rustc_dirty;
pub const ATTR_CLEAN: Symbol = sym::rustc_clean;
pub const ATTR_IF_THIS_CHANGED: Symbol = sym::rustc_if_this_changed;
pub const ATTR_THEN_THIS_WOULD_NEED: Symbol = sym::rustc_then_this_would_need;
pub const ATTR_PARTITION_REUSED: Symbol = sym::rustc_partition_reused;
pub const ATTR_PARTITION_CODEGENED: Symbol = sym::rustc_partition_codegened;
pub const ATTR_EXPECTED_CGU_REUSE: Symbol = sym::rustc_expected_cgu_reuse;

pub const IGNORED_ATTRIBUTES: &[Symbol] = &[
    sym::cfg,
    ATTR_IF_THIS_CHANGED,
    ATTR_THEN_THIS_WOULD_NEED,
    ATTR_DIRTY,
    ATTR_CLEAN,
    ATTR_PARTITION_REUSED,
    ATTR_PARTITION_CODEGENED,
    ATTR_EXPECTED_CGU_REUSE,
];
