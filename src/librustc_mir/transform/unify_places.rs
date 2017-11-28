// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::indexed_set::{IdxSet, IdxSetBuf};
use rustc::hir;
use rustc::mir::*;
use rustc::mir::tcx::PlaceTy;
use rustc::mir::visit::{PlaceContext, MutVisitor, Visitor};
use rustc::session::config::FullDebugInfo;
use rustc::ty::TyCtxt;
use std::mem;
use analysis::accesses::Accesses;
use analysis::eventflow::{After, Before, SparseBitSet};
use analysis::locations::FlatLocations;
use transform::{MirPass, MirSource};

struct Finder<I: Idx> {
    parent: IndexVec<I, I>,
}

impl<I: Idx> Finder<I> {
    fn find(&mut self, i: I) -> I {
        let parent = self.parent[i];
        if i == parent {
            return i;
        }
        let root = self.find(parent);
        if root != parent {
            self.parent[i] = root;
        }
        root
    }
}

/// Union-find for the points of a binary symmetric relation.
/// Note that the relation is not transitive, only `union` is.
struct UnionFindSymRel<I: Idx> {
    finder: Finder<I>,
    relation: IndexVec<I, SparseBitSet<I>>,
}

impl<I: Idx> UnionFindSymRel<I> {
    fn union(&mut self, a: I, b: I) -> I {
        let a = self.finder.find(a);
        let b = self.finder.find(b);
        if a == b {
            return a;
        }

        let (root, child) = if self.relation[a].capacity() > self.relation[b].capacity() {
            (a, b)
        } else {
            (b, a)
        };
        self.finder.parent[child] = root;

        // Have to juggle the `self.relation` elements as we have
        // no way to borrow two disjoint elements at the same time.
        let child_relation = mem::replace(&mut self.relation[child], SparseBitSet::new());
        // FIXME(eddyb) This could use per-"word" bitwise operations.
        for i in child_relation.iter() {
            // HACK(eddyb) this is really expensive, but used to propagate the relation.
            let i = self.finder.find(i);
            self.relation[root].insert(i);
            self.relation[i].insert(root);
        }
        self.relation[child] = child_relation;

        root
    }

    fn relates(&mut self, a: I, b: I) -> bool {
        let a = self.finder.find(a);
        let b = self.finder.find(b);
        self.relation[a].contains(b) || self.relation[b].contains(a)
    }
}

#[derive(Copy, Clone, Debug)]
struct LocalInterior<'a, 'tcx: 'a> {
    base: Local,
    subplace: Option<&'a Place<'tcx>>
}

impl<'a, 'tcx> LocalInterior<'a, 'tcx> {
    fn from_place(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                  mir: &Mir<'tcx>,
                  place: &'a Place<'tcx>)
                  -> Option<(Self, PlaceTy<'tcx>)> {
        match *place {
            Place::Local(base) => {
                Some((LocalInterior { base, subplace: None },
                      PlaceTy::from_ty(mir.local_decls[base].ty)))
            }
            Place::Static(_) => None,
            Place::Projection(ref proj) => {
                let (base, base_ty) = Self::from_place(tcx, mir, &proj.base)?;

                // Packed types can have under-aligned fields, which
                // can't be freely used wherever places are required
                // and/or assumed to be aligned, e.g. safe borrows.
                let adt_def = match base_ty {
                    PlaceTy::Ty { ty } => ty.ty_adt_def(),
                    PlaceTy::Downcast { adt_def, .. } => Some(adt_def)
                };
                if adt_def.map_or(false, |adt| adt.repr.packed()) {
                    return None;
                }

                let ty = match proj.elem {
                    ProjectionElem::Field(..) |
                    ProjectionElem::Downcast(..) |
                    ProjectionElem::ConstantIndex { .. } |
                    ProjectionElem::Subslice { .. } => {
                        base_ty.projection_ty(tcx, &proj.elem)
                    }

                    ProjectionElem::Index(_) |
                    ProjectionElem::Deref => return None
                };
                Some((LocalInterior {
                    base: base.base,
                    subplace: Some(place)
                }, ty))
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct RenameCandidate<'a, 'tcx: 'a> {
    from: Local,
    to: LocalInterior<'a, 'tcx>
}

impl<'a, 'tcx> RenameCandidate<'a, 'tcx> {
    fn new(a: LocalInterior<'a, 'tcx>, b: LocalInterior<'a, 'tcx>) -> Option<Self> {
        // Only locals may be renamed.
        let (from, to) = match (a.subplace, b.subplace) {
            (None, _) => (a.base, b),
            (_, None) => (b.base, a),
            _ => return None
        };
        Some(RenameCandidate { from, to })
    }
}

impl<'a, 'tcx> RenameCandidate<'a, 'tcx> {
    fn filter(self, can_rename: &IdxSet<Local>) -> Option<Self> {
        if self.from == self.to.base {
            return None;
        }

        if can_rename.contains(&self.from) {
            Some(self)
        } else if can_rename.contains(&self.to.base) && self.to.subplace.is_none() {
            Some(RenameCandidate {
                from: self.to.base,
                to: LocalInterior {
                    base: self.from,
                    subplace: None
                }
            })
        } else {
            None
        }
    }
}

pub struct UnifyPlaces;

impl MirPass for UnifyPlaces {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        // Don't run on constant MIR, because trans might not be able to
        // evaluate the modified MIR.
        // FIXME(eddyb) Remove check after miri is merged.
        let id = tcx.hir.as_local_node_id(source.def_id).unwrap();
        match (tcx.hir.body_owner_kind(id), source.promoted) {
            (_, Some(_)) |
            (hir::BodyOwnerKind::Const, _) |
            (hir::BodyOwnerKind::Static(_), _) => return,

            (hir::BodyOwnerKind::Fn, _) => {
                if tcx.is_const_fn(source.def_id) {
                    // Don't run on const functions, as, again, trans might not be able to evaluate
                    // the optimized IR.
                    return
                }
            }
        }

        let mut replacement_place = IndexVec::from_elem(None, &mir.local_decls);
        let mut replacement_finder = Finder {
            parent: mir.local_decls.indices().collect(),
        };
        {
            let can_rename = &mut IdxSetBuf::new_empty(mir.local_decls.len());

            // We need to keep user variables intact for debuginfo.
            // FIXME(eddyb) We should allow multiple user variables
            // per local for debuginfo instead of not optimizing them.
            if tcx.sess.opts.debuginfo == FullDebugInfo {
                for local in mir.temps_iter() {
                    can_rename.add(&local);
                }
            } else {
                // FIXME(eddyb) use ranges for performance.
                for local in mir.vars_and_temps_iter() {
                    can_rename.add(&local);
                }
            }

            {
                let mut can_rename_collector = CanRenameCollector {
                    can_rename
                };
                can_rename_collector.visit_mir(mir);
            }

            let flat_locations = &FlatLocations::collect(mir);
            let accesses = &Accesses::collect(mir, flat_locations);
            let mut observers = accesses.results.observe();
            let mut conflicts = IndexVec::from_elem(SparseBitSet::new(), &mir.local_decls);
            let mut candidates = vec![];
            for (block, data) in mir.basic_blocks().iter_enumerated() {
                let mut add_conflicts_at = |past: &IdxSet<_>, future: &IdxSet<_>| {
                    // FIXME(eddyb) use `diff_at_location` (how?) as an optimization.
                    for i in past.iter() {
                        if future.contains(&i) {
                            // FIXME(eddyb) Reduce the cost of this already Q_Q.
                            for j in past.iter() {
                                if i != j && future.contains(&j) {
                                    conflicts[i].insert(j);
                                }
                            }
                        }
                    }
                };
                let mut checked_after_last_statement = false;
                for (statement_index, stmt) in data.statements.iter().enumerate() {
                    let location = Location { block, statement_index };
                    match stmt.kind {
                        // FIXME(eddyb) figure out how to allow copies.
                        // Maybe if there is any candidate that's a copy,
                        // mark the unification as needing copies? Unclear.
                        // StatementKind::Assign(ref dest, Rvalue::Use(Operand::Copy(ref src))) |
                        StatementKind::Assign(ref dest, Rvalue::Use(Operand::Move(ref src))) => {
                            if let Some((dest, _)) = LocalInterior::from_place(tcx, mir, dest) {
                                if let Some((src, _)) = LocalInterior::from_place(tcx, mir, src) {
                                    candidates.extend(RenameCandidate::new(dest, src)
                                        .and_then(|c| c.filter(can_rename)));
                                    if !checked_after_last_statement {
                                        add_conflicts_at(
                                            observers.past.seek(Before(location)),
                                            observers.future.seek(Before(location)));
                                    }
                                    checked_after_last_statement = false;
                                    continue;
                                }
                            }
                        }
                        _ => {}
                    }
                    add_conflicts_at(
                        observers.past.seek(After(location)),
                        observers.future.seek(Before(location)));
                    checked_after_last_statement = true;
                }
                let location = Location {
                    block,
                    statement_index: data.statements.len()
                };
                add_conflicts_at(
                    observers.past.seek(After(location)),
                    observers.future.seek(Before(location)));
            }

            let mut conflicts = UnionFindSymRel {
                finder: Finder {
                    parent: mir.local_decls.indices().collect(),
                },
                relation: conflicts
            };

            // Union together all the candidate source and targets.
            // Candidates may fail if they could cause a conflict.
            for mut candidate in candidates {
                debug!("unify_places: original: {:?}", candidate);
                candidate.from = replacement_finder.find(candidate.from);
                candidate.to.base = replacement_finder.find(candidate.to.base);
                let candidate = candidate.filter(can_rename);
                debug!("unify_places: filtered: {:?}", candidate);
                if let Some(RenameCandidate { from, to }) = candidate {
                    if conflicts.relates(from, to.base) {
                        continue;
                    }
                    conflicts.union(from, to.base);

                    if let Some(to_place) = to.subplace {
                        debug!("unify_places: {:?} -> {:?}", from, to_place);
                        replacement_place[from] = Some(to_place.clone());
                        can_rename.remove(&from);
                    } else {
                        debug!("unify_places: {:?} -> {:?}", from, to.base);
                        replacement_finder.parent[from] = to.base;
                    }
                }
            }
        }

        // Apply the replacements we computed previously.
        let mut replacer = Replacer {
            replacement_place,
            replacement_finder
        };
        replacer.visit_mir(mir);
    }
}

struct CanRenameCollector<'a> {
    can_rename: &'a mut IdxSet<Local>
}

impl<'a, 'tcx> Visitor<'tcx> for CanRenameCollector<'a> {
    fn visit_projection_elem(&mut self,
                             elem: &PlaceElem<'tcx>,
                             context: PlaceContext<'tcx>,
                             location: Location) {
        if let ProjectionElem::Index(i) = *elem {
            // FIXME(eddyb) We could rename locals used as indices,
            // but only to other whole locals, not their fields.
            self.can_rename.remove(&i);
        }
        self.super_projection_elem(elem, context, location);
    }
}

struct Replacer<'tcx> {
    replacement_place: IndexVec<Local, Option<Place<'tcx>>>,
    replacement_finder: Finder<Local>
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'tcx> {
    fn visit_place(&mut self,
                   place: &mut Place<'tcx>,
                   context: PlaceContext<'tcx>,
                   location: Location) {
        if let Place::Local(from) = *place {
            let to = self.replacement_finder.find(from);
            let to_place = self.replacement_place[to].as_ref().cloned();
            if let Some(to_place) = to_place {
                *place = to_place;
            } else if to != from {
                *place = Place::Local(to);
            }

            // Recurse in case the replacement also needs to be replaced.
            // FIXME(eddyb) precompute it into `replacement_place`.
        }

        self.super_place(place, context, location);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        // FIXME(eddyb) fuse storage liveness ranges instead of removing them.
        match statement.kind {
            StatementKind::StorageLive(_local) |
            StatementKind::StorageDead(_local) => {
                // FIXME(eddyb) figure out how to even detect relevancy.
                statement.make_nop();
            }
            _ => {}
        }

        self.super_statement(block, statement, location);

        // Remove self-assignments resulting from replaced move chains.
        let nop = match statement.kind {
            StatementKind::Assign(ref dest, Rvalue::Use(Operand::Copy(ref src))) |
            StatementKind::Assign(ref dest, Rvalue::Use(Operand::Move(ref src))) => {
                dest == src
            }
            _ => false
        };
        // HACK(eddyb) clean this up post-NLL.
        if nop {
            statement.make_nop();
        }
    }
}
