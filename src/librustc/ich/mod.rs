// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::fingerprint::Fingerprint;
pub use self::def_path_hash::DefPathHashes;
pub use self::caching_codemap_view::CachingCodemapView;

mod fingerprint;
mod def_path_hash;
mod caching_codemap_view;

pub const ATTR_DIRTY: &'static str = "rustc_dirty";
pub const ATTR_CLEAN: &'static str = "rustc_clean";
pub const ATTR_DIRTY_METADATA: &'static str = "rustc_metadata_dirty";
pub const ATTR_CLEAN_METADATA: &'static str = "rustc_metadata_clean";
pub const ATTR_IF_THIS_CHANGED: &'static str = "rustc_if_this_changed";
pub const ATTR_THEN_THIS_WOULD_NEED: &'static str = "rustc_then_this_would_need";

pub const IGNORED_ATTRIBUTES: &'static [&'static str] = &[
    "cfg",
    ATTR_IF_THIS_CHANGED,
    ATTR_THEN_THIS_WOULD_NEED,
    ATTR_DIRTY,
    ATTR_CLEAN,
    ATTR_DIRTY_METADATA,
    ATTR_CLEAN_METADATA
];
