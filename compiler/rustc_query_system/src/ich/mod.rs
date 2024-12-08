//! ICH - Incremental Compilation Hash

use rustc_span::symbol::{Symbol, sym};

pub use self::hcx::StableHashingContext;

mod hcx;
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
