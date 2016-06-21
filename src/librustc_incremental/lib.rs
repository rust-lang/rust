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
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(rustc_private)]
#![feature(staged_api)]

extern crate graphviz;
extern crate rbml;
#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate serialize as rustc_serialize;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;

mod assert_dep_graph;
mod calculate_svh;
mod persist;

pub use assert_dep_graph::assert_dep_graph;
pub use calculate_svh::SvhCalculate;
pub use persist::load_dep_graph;
pub use persist::save_dep_graph;
