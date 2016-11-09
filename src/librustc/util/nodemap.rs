// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An efficient hash map for node IDs

#![allow(non_snake_case)]

use hir::def_id::DefId;
use syntax::ast;

pub use rustc_data_structures::fx::FxHashMap;
pub use rustc_data_structures::fx::FxHashSet;

pub type NodeMap<T> = FxHashMap<ast::NodeId, T>;
pub type DefIdMap<T> = FxHashMap<DefId, T>;

pub type NodeSet = FxHashSet<ast::NodeId>;
pub type DefIdSet = FxHashSet<DefId>;

pub fn NodeMap<T>() -> NodeMap<T> { FxHashMap() }
pub fn DefIdMap<T>() -> DefIdMap<T> { FxHashMap() }
pub fn NodeSet() -> NodeSet { FxHashSet() }
pub fn DefIdSet() -> DefIdSet { FxHashSet() }

