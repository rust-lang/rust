// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An analysis to determine which locals require allocas and
//! which do not.

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::control_flow_graph::dominators::Dominators;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc::mir::{self, Mir, Location};
use rustc::mir::visit::{Visitor, PlaceContext};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable};
use rustc::ty::subst::Substs;
use rustc::ty::layout::{TyLayout, LayoutError};

pub trait LocalAnalyzerCallbacks<'tcx> {
    fn ty_ssa_allowed(&self, ty: Ty<'tcx>) -> bool;
    fn does_rvalue_create_operand(&self, rval: &mir::Rvalue<'tcx>) -> bool;
}

pub fn non_ssa_locals<'a, 'tcx: 'a, C: LocalAnalyzerCallbacks<'tcx>>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    param_substs: &'tcx Substs<'tcx>,
    callbacks: C,
) -> BitVector {
    let mut analyzer = LocalAnalyzer::new(tcx, mir, param_substs, callbacks);

    analyzer.visit_mir(mir);

    for (index, ty) in mir.local_decls.iter().map(|l| l.ty).enumerate() {
        let ty = analyzer.monomorphize(&ty);
        debug!("local {} has type {:?}", index, ty);
        if !analyzer.callbacks.ty_ssa_allowed(ty) {
            analyzer.not_ssa(mir::Local::new(index));
        }
    }

    analyzer.non_ssa_locals
}

struct LocalAnalyzer<'a, 'tcx: 'a, C: LocalAnalyzerCallbacks<'tcx>> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    param_substs: &'tcx Substs<'tcx>,
    callbacks: C,
    dominators: Dominators<mir::BasicBlock>,
    non_ssa_locals: BitVector,
    // The location of the first visited direct assignment to each
    // local, or an invalid location (out of bounds `block` index).
    first_assignment: IndexVec<mir::Local, Location>
}

impl<'a, 'tcx: 'a, C: LocalAnalyzerCallbacks<'tcx>> LocalAnalyzer<'a, 'tcx, C> {
    fn new(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        mir: &'a Mir<'tcx>,
        param_substs: &'tcx Substs<'tcx>,
        callbacks: C,
    ) -> LocalAnalyzer<'a, 'tcx, C> {
        let invalid_location =
            mir::BasicBlock::new(mir.basic_blocks().len()).start_location();
        let mut analyzer = LocalAnalyzer {
            tcx,
            mir,
            param_substs,
            callbacks,
            dominators: mir.dominators(),
            non_ssa_locals: BitVector::new(mir.local_decls.len()),
            first_assignment: IndexVec::from_elem(invalid_location, &mir.local_decls)
        };

        // Arguments get assigned to by means of the function being called
        for arg in mir.args_iter() {
            analyzer.first_assignment[arg] = mir::START_BLOCK.start_location();
        }

        analyzer
    }

    fn first_assignment(&self, local: mir::Local) -> Option<Location> {
        let location = self.first_assignment[local];
        if location.block.index() < self.mir.basic_blocks().len() {
            Some(location)
        } else {
            None
        }
    }

    fn not_ssa(&mut self, local: mir::Local) {
        debug!("marking {:?} as non-SSA", local);
        self.non_ssa_locals.insert(local.index());
    }

    fn assign(&mut self, local: mir::Local, location: Location) {
        if self.first_assignment(local).is_some() {
            self.not_ssa(local);
        } else {
            self.first_assignment[local] = location;
        }
    }

    fn layout_of(&self, ty: Ty<'tcx>) -> TyLayout<'tcx> {
        self.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty))
            .unwrap_or_else(|e| match e {
                LayoutError::SizeOverflow(_) => self.tcx.sess.fatal(&e.to_string()),
                _ => bug!("failed to get layout for `{}`: {}", ty, e)
            })
    }

    fn monomorphize<T>(&self, value: &T) -> T
        where T: TypeFoldable<'tcx>
    {
        self.tcx.subst_and_normalize_erasing_regions(
            self.param_substs,
            ty::ParamEnv::reveal_all(),
            value,
        )
    }
}

impl<'a, 'tcx, C: LocalAnalyzerCallbacks<'tcx>> Visitor<'tcx> for LocalAnalyzer<'a, 'tcx, C> {
    fn visit_assign(&mut self,
                    block: mir::BasicBlock,
                    place: &mir::Place<'tcx>,
                    rvalue: &mir::Rvalue<'tcx>,
                    location: Location) {
        debug!("visit_assign(block={:?}, place={:?}, rvalue={:?})", block, place, rvalue);

        if let mir::Place::Local(index) = *place {
            self.assign(index, location);
            if !self.callbacks.does_rvalue_create_operand(rvalue) {
                self.not_ssa(index);
            }
        } else {
            self.visit_place(place, PlaceContext::Store, location);
        }

        self.visit_rvalue(rvalue, location);
    }

    fn visit_terminator_kind(&mut self,
                             block: mir::BasicBlock,
                             kind: &mir::TerminatorKind<'tcx>,
                             location: Location) {
        let check = match *kind {
            mir::TerminatorKind::Call {
                func: mir::Operand::Constant(ref c),
                ref args, ..
            } => match c.ty.sty {
                ty::TyFnDef(did, _) => Some((did, args)),
                _ => None,
            },
            _ => None,
        };
        if let Some((def_id, args)) = check {
            if Some(def_id) == self.tcx.lang_items().box_free_fn() {
                // box_free(x) shares with `drop x` the property that it
                // is not guaranteed to be statically dominated by the
                // definition of x, so x must always be in an alloca.
                if let mir::Operand::Move(ref place) = args[0] {
                    self.visit_place(place, PlaceContext::Drop, location);
                }
            }
        }

        self.super_terminator_kind(block, kind, location);
    }

    fn visit_place(&mut self,
                    place: &mir::Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        debug!("visit_place(place={:?}, context={:?})", place, context);
        if let mir::Place::Projection(ref proj) = *place {
            // Allow uses of projections that are ZSTs or from scalar fields.
            let is_consume = match context {
                PlaceContext::Copy | PlaceContext::Move => true,
                _ => false
            };
            if is_consume {
                let base_ty = proj.base.ty(self.mir, self.tcx);
                let base_ty = self.monomorphize(&base_ty);

                // ZSTs don't require any actual memory access.
                let elem_ty = base_ty.projection_ty(self.tcx, &proj.elem).to_ty(self.tcx);
                let elem_ty = self.monomorphize(&elem_ty);
                if self.layout_of(elem_ty).is_zst() {
                    return;
                }

                if let mir::ProjectionElem::Field(..) = proj.elem {
                    if self.callbacks.ty_ssa_allowed(base_ty.to_ty(self.tcx)) {
                        // Recurse with the same context, instead of `Projection`,
                        // potentially stopping at non-operand projections,
                        // which would trigger `not_ssa` on locals.
                        self.visit_place(&proj.base, context, location);
                        return;
                    }
                }
            }

            // A deref projection only reads the pointer, never needs the place.
            if let mir::ProjectionElem::Deref = proj.elem {
                return self.visit_place(&proj.base, PlaceContext::Copy, location);
            }
        }

        self.super_place(place, context, location);
    }

    fn visit_local(&mut self,
                   &local: &mir::Local,
                   context: PlaceContext<'tcx>,
                   location: Location) {
        match context {
            PlaceContext::Call => {
                self.assign(local, location);
            }

            PlaceContext::StorageLive |
            PlaceContext::StorageDead |
            PlaceContext::Validate => {}

            PlaceContext::Copy |
            PlaceContext::Move => {
                // Reads from uninitialized variables (e.g. in dead code, after
                // optimizations) require locals to be in (uninitialized) memory.
                // NB: there can be uninitialized reads of a local visited after
                // an assignment to that local, if they happen on disjoint paths.
                let ssa_read = match self.first_assignment(local) {
                    Some(assignment_location) => {
                        assignment_location.dominates(location, &self.dominators)
                    }
                    None => false
                };
                if !ssa_read {
                    self.not_ssa(local);
                }
            }

            PlaceContext::Inspect |
            PlaceContext::Store |
            PlaceContext::AsmOutput |
            PlaceContext::Borrow { .. } |
            PlaceContext::Projection(..) => {
                self.not_ssa(local);
            }

            PlaceContext::Drop => {
                let ty = mir::Place::Local(local).ty(self.mir, self.tcx);
                let ty = self.monomorphize(&ty.to_ty(self.tcx));

                // Only need the place if we're actually dropping it.
                if ty.needs_drop(self.tcx, ty::ParamEnv::reveal_all()) {
                    self.not_ssa(local);
                }
            }
        }
    }
}
