//! ICH - Incremental Compilation Hash

pub use self::hcx::{NodeIdHashingMode, StableHashingContext};
use rustc_span::symbol::{sym, Symbol};

mod hcx;
mod impls_hir;
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
