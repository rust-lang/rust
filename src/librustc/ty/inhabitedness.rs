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
use syntax::ast::CRATE_NODE_ID;
use util::nodemap::FxHashSet;
use ty::context::TyCtxt;
use ty::{AdtDef, VariantDef, FieldDef, TyS};
use ty::{DefId, Substs};
use ty::{AdtKind, Visibility, DefIdTree};
use ty::TypeVariants::*;

/// Represents a set of DefIds closed under the ancestor relation. That is, if
/// a DefId is in this set then so are all its descendants.
#[derive(Clone)]
pub struct DefIdForest {
    /// The minimal set of DefIds required to represent the whole set.
    /// If A and B are DefIds in the DefIdForest, and A is a desecendant
    /// of B, then only B will be in root_ids.
    /// We use a SmallVec here because (for its use in this module) its rare
    /// that this will contain even two ids.
    root_ids: SmallVec<[DefId; 1]>,
}

impl<'a, 'gcx, 'tcx> DefIdForest {
    /// Create an empty forest.
    pub fn empty() -> DefIdForest {
        DefIdForest {
            root_ids: SmallVec::new(),
        }
    }

    /// Create a forest consisting of a single tree representing the entire
    /// crate.
    #[inline]
    pub fn full(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest {
        let crate_id = tcx.map.local_def_id(CRATE_NODE_ID);
        DefIdForest::from_id(crate_id)
    }

    /// Create a forest containing a DefId and all its descendants.
    pub fn from_id(id: DefId) -> DefIdForest {
        let mut root_ids = SmallVec::new();
        root_ids.push(id);
        DefIdForest {
            root_ids: root_ids,
        }
    }

    /// Test whether the forest is empty.
    pub fn is_empty(&self) -> bool {
        self.root_ids.is_empty()
    }

    /// Test whether the forest conains a given DefId.
    pub fn contains(&self,
                    tcx: TyCtxt<'a, 'gcx, 'tcx>,
                    id: DefId) -> bool
    {
        for root_id in self.root_ids.iter() {
            if tcx.is_descendant_of(id, *root_id) {
                return true;
            }
        }
        false
    }

    /// Calculate the intersection of a collection of forests.
    pub fn intersection<I>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                           iter: I) -> DefIdForest
            where I: IntoIterator<Item=DefIdForest>
    {
        let mut ret = DefIdForest::full(tcx);
        let mut next_ret = SmallVec::new();
        let mut old_ret: SmallVec<[DefId; 1]> = SmallVec::new();
        for next_forest in iter {
            for id in ret.root_ids.drain(..) {
                if next_forest.contains(tcx, id) {
                    next_ret.push(id);
                } else {
                    old_ret.push(id);
                }
            }
            ret.root_ids.extend(old_ret.drain(..));

            for id in next_forest.root_ids {
                if ret.contains(tcx, id) {
                    next_ret.push(id);
                }
            }

            mem::swap(&mut next_ret, &mut ret.root_ids);
            next_ret.drain(..);
        }
        ret
    }

    /// Calculate the union of a collection of forests.
    pub fn union<I>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                    iter: I) -> DefIdForest
            where I: IntoIterator<Item=DefIdForest>
    {
        let mut ret = DefIdForest::empty();
        let mut next_ret = SmallVec::new();
        for next_forest in iter {
            for id in ret.root_ids.drain(..) {
                if !next_forest.contains(tcx, id) {
                    next_ret.push(id);
                }
            }

            for id in next_forest.root_ids {
                if !next_ret.contains(&id) {
                    next_ret.push(id);
                }
            }

            mem::swap(&mut next_ret, &mut ret.root_ids);
            next_ret.drain(..);
        }
        ret
    }
}

impl<'a, 'gcx, 'tcx> AdtDef {
    /// Calculate the forest of DefIds from which this adt is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>) -> DefIdForest
    {
        if !visited.insert((self.did, substs)) {
            return DefIdForest::empty();
        }

        let ret = DefIdForest::intersection(tcx, self.variants.iter().map(|v| {
            v.uninhabited_from(visited, tcx, substs, self.adt_kind())
        }));
        visited.remove(&(self.did, substs));
        ret
    }
}

impl<'a, 'gcx, 'tcx> VariantDef {
    /// Calculate the forest of DefIds from which this variant is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>,
                adt_kind: AdtKind) -> DefIdForest
    {
        match adt_kind {
            AdtKind::Union => {
                DefIdForest::intersection(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, false)
                }))
            },
            AdtKind::Struct => {
                DefIdForest::union(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, false)
                }))
            },
            AdtKind::Enum => {
                DefIdForest::union(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, true)
                }))
            },
        }
    }
}

impl<'a, 'gcx, 'tcx> FieldDef {
    /// Calculate the forest of DefIds from which this field is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>,
                is_enum: bool) -> DefIdForest
    {
        let mut data_uninhabitedness = move || self.ty(tcx, substs).uninhabited_from(visited, tcx);
        if is_enum {
            data_uninhabitedness()
        } else {
            match self.vis {
                Visibility::Invisible => DefIdForest::empty(),
                Visibility::Restricted(from) => {
                    let forest = DefIdForest::from_id(from);
                    let iter = Some(forest).into_iter().chain(Some(data_uninhabitedness()));
                    DefIdForest::intersection(tcx, iter)
                },
                Visibility::Public => data_uninhabitedness(),
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> TyS<'tcx> {
    /// Calculate the forest of DefIds from which this type is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest
    {
        match tcx.lift_to_global(&self) {
            Some(global_ty) => {
                {
                    let cache = tcx.inhabitedness_cache.borrow();
                    if let Some(forest) = cache.get(&global_ty) {
                        return forest.clone();
                    }
                }
                let forest = global_ty.uninhabited_from_inner(visited, tcx);
                let mut cache = tcx.inhabitedness_cache.borrow_mut();
                cache.insert(global_ty, forest.clone());
                forest
            },
            None => {
                let forest = self.uninhabited_from_inner(visited, tcx);
                forest
            },
        }
    }

    fn uninhabited_from_inner(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest
    {
        match self.sty {
            TyAdt(def, substs) => {
                def.uninhabited_from(visited, tcx, substs)
            },

            TyNever => DefIdForest::full(tcx),
            TyTuple(ref tys) => {
                DefIdForest::union(tcx, tys.iter().map(|ty| {
                    ty.uninhabited_from(visited, tcx)
                }))
            },
            TyArray(ty, len) => {
                if len == 0 {
                    DefIdForest::empty()
                } else {
                    ty.uninhabited_from(visited, tcx)
                }
            }
            TyRef(_, ref tm) => tm.ty.uninhabited_from(visited, tcx),

            _ => DefIdForest::empty(),
        }
    }
}

