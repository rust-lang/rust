// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See [the README](README.md) for details on writing your own pass.

use hir;
use hir::def_id::DefId;
use hir::map::DefPathData;
use mir::Promoted;
use ty::TyCtxt;
use syntax::ast::NodeId;

/// Where a specific Mir comes from.
#[derive(Debug, Copy, Clone)]
pub enum MirSource {
    /// Functions and methods.
    Fn(NodeId),

    /// Constants and associated constants.
    Const(NodeId),

    /// Initializer of a `static` item.
    Static(NodeId, hir::Mutability),

    /// Promoted rvalues within a function.
    Promoted(NodeId, Promoted),
}

impl<'a, 'gcx, 'tcx> MirSource {
    pub fn from_local_def_id(tcx: TyCtxt<'a, 'gcx, 'tcx>, def_id: DefId) -> MirSource {
        let id = tcx.hir.as_local_node_id(def_id).expect("mir source requires local def-id");
        Self::from_node(tcx, id)
    }

    pub fn from_node(tcx: TyCtxt<'a, 'gcx, 'tcx>, id: NodeId) -> MirSource {
        use hir::*;

        // Handle constants in enum discriminants, types, and repeat expressions.
        let def_id = tcx.hir.local_def_id(id);
        let def_key = tcx.def_key(def_id);
        if def_key.disambiguated_data.data == DefPathData::Initializer {
            return MirSource::Const(id);
        }

        match tcx.hir.get(id) {
            map::NodeItem(&Item { node: ItemConst(..), .. }) |
            map::NodeTraitItem(&TraitItem { node: TraitItemKind::Const(..), .. }) |
            map::NodeImplItem(&ImplItem { node: ImplItemKind::Const(..), .. }) => {
                MirSource::Const(id)
            }
            map::NodeItem(&Item { node: ItemStatic(_, m, _), .. }) => {
                MirSource::Static(id, m)
            }
            // Default to function if it's not a constant or static.
            _ => MirSource::Fn(id)
        }
    }

    pub fn item_id(&self) -> NodeId {
        match *self {
            MirSource::Fn(id) |
            MirSource::Const(id) |
            MirSource::Static(id, _) |
            MirSource::Promoted(id, _) => id
        }
    }
}
