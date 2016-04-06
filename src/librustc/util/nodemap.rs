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

pub use rustc_data_structures::fnv::FnvHashMap;
pub use rustc_data_structures::fnv::FnvHashSet;

pub type NodeMap<T> = FnvHashMap<ast::NodeId, T>;
pub type DefIdMap<T> = FnvHashMap<DefId, T>;

pub type NodeSet = FnvHashSet<ast::NodeId>;
pub type DefIdSet = FnvHashSet<DefId>;

pub fn NodeMap<T>() -> NodeMap<T> { FnvHashMap() }
pub fn DefIdMap<T>() -> DefIdMap<T> { FnvHashMap() }
pub fn NodeSet() -> NodeSet { FnvHashSet() }
pub fn DefIdSet() -> DefIdSet { FnvHashSet() }

