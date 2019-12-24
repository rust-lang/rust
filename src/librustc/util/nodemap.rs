//! An efficient hash map for `NodeId`s.

use crate::hir::{HirId, ItemLocalId};

use rustc_data_structures::define_id_collections;

define_id_collections!(HirIdMap, HirIdSet, HirId);
define_id_collections!(ItemLocalMap, ItemLocalSet, ItemLocalId);
