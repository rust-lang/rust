// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use rustc_data_structures::small_vec::SmallVec;
use syntax::ast::{CRATE_NODE_ID, NodeId};
use util::nodemap::FxHashSet;
use ty::context::TyCtxt;
use ty::{AdtDef, VariantDef, FieldDef, TyS};
use ty::{DefId, Substs};
use ty::{AdtKind, Visibility, NodeIdTree};
use ty::TypeVariants::*;

/// Represents a set of nodes closed under the ancestor relation. That is, if a
/// node is in this set then so are all its descendants.
#[derive(Clone)]
pub struct NodeForrest {
    /// The minimal set of nodes required to represent the whole set.
    /// If A and B are nodes in the NodeForrest, and A is a desecendant
    /// of B, then only B will be in root_nodes.
    /// We use a SmallVec here because (for its use in this module) its rare
    /// that this will contain more than one or two nodes.
    root_nodes: SmallVec<[NodeId; 1]>,
}

impl<'a, 'gcx, 'tcx> NodeForrest {
    /// Create an empty set.
    pub fn empty() -> NodeForrest {
        NodeForrest {
            root_nodes: SmallVec::new(),
        }
    }

    /// Create a set containing every node.
    #[inline]
    pub fn full() -> NodeForrest {
        NodeForrest::from_node(CRATE_NODE_ID)
    }

    /// Create a set containing a node and all its descendants.
    pub fn from_node(node: NodeId) -> NodeForrest {
        let mut root_nodes = SmallVec::new();
        root_nodes.push(node);
        NodeForrest {
            root_nodes: root_nodes,
        }
    }

    /// Test whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.root_nodes.is_empty()
    }

    /// Test whether the set conains a node.
    pub fn contains(&self,
                    tcx: TyCtxt<'a, 'gcx, 'tcx>,
                    node: NodeId) -> bool
    {
        for root_node in self.root_nodes.iter() {
            if tcx.map.is_descendant_of(node, *root_node) {
                return true;
            }
        }
        false
    }

    /// Calculate the intersection of a collection of sets.
    pub fn intersection<I>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                           iter: I) -> NodeForrest
            where I: IntoIterator<Item=NodeForrest>
    {
        let mut ret = NodeForrest::full();
        let mut next_ret = SmallVec::new();
        let mut old_ret: SmallVec<[NodeId; 1]> = SmallVec::new();
        for next_set in iter {
            for node in ret.root_nodes.drain(..) {
                if next_set.contains(tcx, node) {
                    next_ret.push(node);
                } else {
                    old_ret.push(node);
                }
            }
            ret.root_nodes.extend(old_ret.drain(..));

            for node in next_set.root_nodes {
                if ret.contains(tcx, node) {
                    next_ret.push(node);
                }
            }

            mem::swap(&mut next_ret, &mut ret.root_nodes);
            next_ret.drain(..);
        }
        ret
    }

    /// Calculate the union of a collection of sets.
    pub fn union<I>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                    iter: I) -> NodeForrest
            where I: IntoIterator<Item=NodeForrest>
    {
        let mut ret = NodeForrest::empty();
        let mut next_ret = SmallVec::new();
        for next_set in iter {
            for node in ret.root_nodes.drain(..) {
                if !next_set.contains(tcx, node) {
                    next_ret.push(node);
                }
            }

            for node in next_set.root_nodes {
                if !next_ret.contains(&node) {
                    next_ret.push(node);
                }
            }

            mem::swap(&mut next_ret, &mut ret.root_nodes);
            next_ret.drain(..);
        }
        ret
    }
}

impl<'a, 'gcx, 'tcx> AdtDef {
    /// Calculate the set of  nodes from which this adt is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>) -> NodeForrest
    {
        if !visited.insert((self.did, substs)) {
            return NodeForrest::empty();
        }

        let ret = NodeForrest::intersection(tcx, self.variants.iter().map(|v| {
            v.uninhabited_from(visited, tcx, substs, self.adt_kind())
        }));
        visited.remove(&(self.did, substs));
        ret
    }
}

impl<'a, 'gcx, 'tcx> VariantDef {
    /// Calculate the set of  nodes from which this variant is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>,
                adt_kind: AdtKind) -> NodeForrest
    {
        match adt_kind {
            AdtKind::Union => {
                NodeForrest::intersection(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, false)
                }))
            },
            AdtKind::Struct => {
                NodeForrest::union(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, false)
                }))
            },
            AdtKind::Enum => {
                NodeForrest::union(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, true)
                }))
            },
        }
    }
}

impl<'a, 'gcx, 'tcx> FieldDef {
    /// Calculate the set of  nodes from which this field is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>,
                is_enum: bool) -> NodeForrest
    {
        let mut data_uninhabitedness = move || self.ty(tcx, substs).uninhabited_from(visited, tcx);
        if is_enum {
            data_uninhabitedness()
        } else {
            match self.vis {
                Visibility::PrivateExternal => NodeForrest::empty(),
                Visibility::Restricted(from) => {
                    let node_set = NodeForrest::from_node(from);
                    let iter = Some(node_set).into_iter().chain(Some(data_uninhabitedness()));
                    NodeForrest::intersection(tcx, iter)
                },
                Visibility::Public => data_uninhabitedness(),
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> TyS<'tcx> {
    /// Calculate the set of  nodes from which this type is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>) -> NodeForrest
    {
        match tcx.lift_to_global(&self) {
            Some(global_ty) => {
                {
                    let cache = tcx.inhabitedness_cache.borrow();
                    if let Some(closed_node_set) = cache.get(&global_ty) {
                        return closed_node_set.clone();
                    }
                }
                let node_set = global_ty.uninhabited_from_inner(visited, tcx);
                let mut cache = tcx.inhabitedness_cache.borrow_mut();
                cache.insert(global_ty, node_set.clone());
                node_set
            },
            None => {
                let node_set = self.uninhabited_from_inner(visited, tcx);
                node_set
            },
        }
    }

    fn uninhabited_from_inner(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>) -> NodeForrest
    {
        match self.sty {
            TyAdt(def, substs) => {
                def.uninhabited_from(visited, tcx, substs)
            },

            TyNever => NodeForrest::full(),
            TyTuple(ref tys) => {
                NodeForrest::union(tcx, tys.iter().map(|ty| {
                    ty.uninhabited_from(visited, tcx)
                }))
            },
            TyArray(ty, len) => {
                if len == 0 {
                    NodeForrest::empty()
                } else {
                    ty.uninhabited_from(visited, tcx)
                }
            }
            TyRef(_, ref tm) => tm.ty.uninhabited_from(visited, tcx),

            _ => NodeForrest::empty(),
        }
    }
}

