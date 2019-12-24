//! An efficient hash map for `NodeId`s.

use crate::hir::def_id::DefId;
use crate::hir::{HirId, ItemLocalId};
use syntax::ast;

use rustc_data_structures::define_id_collections;

define_id_collections!(NodeMap, NodeSet, ast::NodeId);
define_id_collections!(DefIdMap, DefIdSet, DefId);
define_id_collections!(HirIdMap, HirIdSet, HirId);
define_id_collections!(ItemLocalMap, ItemLocalSet, ItemLocalId);
