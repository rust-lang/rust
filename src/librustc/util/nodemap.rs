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
use hir::{HirId, ItemLocalId};
use syntax::ast;

pub use rustc_data_structures::fx::FxHashMap;
pub use rustc_data_structures::fx::FxHashSet;

macro_rules! define_id_collections {
    ($map_name:ident, $set_name:ident, $key:ty) => {
        pub type $map_name<T> = FxHashMap<$key, T>;
        pub fn $map_name<T>() -> $map_name<T> { FxHashMap() }
        pub type $set_name = FxHashSet<$key>;
        pub fn $set_name() -> $set_name { FxHashSet() }
    }
}

define_id_collections!(NodeMap, NodeSet, ast::NodeId);
define_id_collections!(DefIdMap, DefIdSet, DefId);
define_id_collections!(HirIdMap, HirIdSet, HirId);
define_id_collections!(ItemLocalMap, ItemLocalSet, ItemLocalId);
