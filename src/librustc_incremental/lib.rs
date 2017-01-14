// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support for serializing the dep-graph and reloading it.

#![crate_name = "rustc_incremental"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(rand)]
#![feature(core_intrinsics)]

extern crate graphviz;
#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate serialize as rustc_serialize;

#[macro_use] extern crate log;
extern crate syntax;
extern crate syntax_pos;

const ATTR_DIRTY: &'static str = "rustc_dirty";
const ATTR_CLEAN: &'static str = "rustc_clean";
const ATTR_DIRTY_METADATA: &'static str = "rustc_metadata_dirty";
const ATTR_CLEAN_METADATA: &'static str = "rustc_metadata_clean";
const ATTR_IF_THIS_CHANGED: &'static str = "rustc_if_this_changed";
const ATTR_THEN_THIS_WOULD_NEED: &'static str = "rustc_then_this_would_need";

mod assert_dep_graph;
mod calculate_svh;
mod persist;
pub mod ich;

pub use assert_dep_graph::assert_dep_graph;
pub use calculate_svh::compute_incremental_hashes_map;
pub use calculate_svh::IncrementalHashesMap;
pub use calculate_svh::IchHasher;
pub use persist::load_dep_graph;
pub use persist::save_dep_graph;
pub use persist::save_trans_partition;
pub use persist::save_work_products;
pub use persist::in_incr_comp_dir;
pub use persist::finalize_session_directory;
pub use persist::delete_workproduct_files;
