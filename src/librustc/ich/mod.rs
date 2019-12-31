//! ICH - Incremental Compilation Hash

pub use self::hcx::{
    hash_stable_trait_impls, NodeIdHashingMode, StableHashingContext, StableHashingContextProvider,
};
crate use rustc_data_structures::fingerprint::Fingerprint;
use syntax::symbol::{sym, Symbol};
pub use syntax_pos::CachingSourceMapView;

mod hcx;

mod impls_hir;
mod impls_syntax;
mod impls_ty;

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
