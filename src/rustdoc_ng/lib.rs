// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "rustdoc_ng",
       vers = "0.1.0",
       uuid = "8c6e4598-1596-4aa5-a24c-b811914bbbc6")];
#[desc = "rustdoc, the Rust documentation extractor"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

#[deny(warnings)];

extern mod syntax;
extern mod rustc;

extern mod extra;

use extra::serialize::Encodable;

pub mod core;
pub mod doctree;
pub mod clean;
pub mod visit_ast;
pub mod fold;
pub mod plugins;
pub mod passes;

pub static SCHEMA_VERSION: &'static str = "0.8.0";

pub static ctxtkey: std::local_data::Key<@core::DocContext> = &std::local_data::Key;
