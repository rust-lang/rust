//! An efficient hash map for `NodeId`s.

use crate::hir::def_id::DefId;
use crate::hir::{HirId, ItemLocalId};
use syntax::ast;

macro_rules! define_id_collections {
    ($map_name:ident, $set_name:ident, $key:ty) => {
        pub type $map_name<T> = rustc_data_structures::fx::FxHashMap<$key, T>;
        pub type $set_name = rustc_data_structures::fx::FxHashSet<$key>;
    };
}

define_id_collections!(NodeMap, NodeSet, ast::NodeId);
define_id_collections!(DefIdMap, DefIdSet, DefId);
define_id_collections!(HirIdMap, HirIdSet, HirId);
define_id_collections!(ItemLocalMap, ItemLocalSet, ItemLocalId);
