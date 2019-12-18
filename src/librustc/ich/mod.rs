//! ICH - Incremental Compilation Hash

crate use rustc_data_structures::fingerprint::Fingerprint;
pub use syntax_pos::CachingSourceMapView;
pub use self::hcx::{StableHashingContextProvider, StableHashingContext, NodeIdHashingMode,
                    hash_stable_trait_impls};
use syntax::symbol::{Symbol, sym};

mod hcx;

mod impls_hir;
mod impls_ty;
mod impls_syntax;

pub const IGNORED_ATTRIBUTES: &[Symbol] = &[
    sym::cfg,
    sym::rustc_if_this_changed,
    sym::rustc_then_this_would_need,
    sym::rustc_dirty,
    sym::rustc_clean,
    sym::rustc_partition_reused,
    sym::rustc_partition_codegened,
    sym::rustc_expected_cgu_reuse,
];
