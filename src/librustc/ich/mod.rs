// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ICH - Incremental Compilation Hash

pub use self::fingerprint::Fingerprint;
pub use self::caching_codemap_view::CachingCodemapView;
pub use self::hcx::{StableHashingContext, NodeIdHashingMode,
                    hash_stable_trait_impls};
mod fingerprint;
mod caching_codemap_view;
mod hcx;

mod impls_const_math;
mod impls_cstore;
mod impls_hir;
mod impls_mir;
mod impls_misc;
mod impls_ty;
mod impls_syntax;

pub const ATTR_DIRTY: &'static str = "rustc_dirty";
pub const ATTR_CLEAN: &'static str = "rustc_clean";
pub const ATTR_DIRTY_METADATA: &'static str = "rustc_metadata_dirty";
pub const ATTR_CLEAN_METADATA: &'static str = "rustc_metadata_clean";
pub const ATTR_IF_THIS_CHANGED: &'static str = "rustc_if_this_changed";
pub const ATTR_THEN_THIS_WOULD_NEED: &'static str = "rustc_then_this_would_need";
pub const ATTR_PARTITION_REUSED: &'static str = "rustc_partition_reused";
pub const ATTR_PARTITION_TRANSLATED: &'static str = "rustc_partition_translated";


pub const DEP_GRAPH_ASSERT_ATTRS: &'static [&'static str] = &[
    ATTR_IF_THIS_CHANGED,
    ATTR_THEN_THIS_WOULD_NEED,
    ATTR_DIRTY,
    ATTR_CLEAN,
    ATTR_DIRTY_METADATA,
    ATTR_CLEAN_METADATA,
    ATTR_PARTITION_REUSED,
    ATTR_PARTITION_TRANSLATED,
];

pub const IGNORED_ATTRIBUTES: &'static [&'static str] = &[
    "cfg",
    ATTR_IF_THIS_CHANGED,
    ATTR_THEN_THIS_WOULD_NEED,
    ATTR_DIRTY,
    ATTR_CLEAN,
    ATTR_DIRTY_METADATA,
    ATTR_CLEAN_METADATA,
    ATTR_PARTITION_REUSED,
    ATTR_PARTITION_TRANSLATED,
];
