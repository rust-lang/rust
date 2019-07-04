//! An analysis to determine which locals require allocas and
//! which do not.

use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc::mir::{self, Location, TerminatorKind};
use rustc::mir::visit::{Visitor, PlaceContext, MutatingUseContext, NonMutatingUseContext};
use rustc::mir::traversal;
use rustc::ty;
use rustc::ty::layout::{LayoutOf, HasTyCtxt};
use super::FunctionCx;
use crate::traits::*;

pub fn non_ssa_locals<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    fx: &FunctionCx<'a, 'tcx, Bx>,
) -> BitSet<mir::Local> {
    let mir = fx.mir;
    let mut analyzer = LocalAnalyzer::new(fx);

    analyzer.visit_body(mir);

    for (index, ty) in mir.local_decls.iter().map(|l| l.ty).enumerate() {
        let ty = fx.monomorphize(&ty);
        debug!("local {} has type {:?}", index, ty);
        let layout = fx.cx.layout_of(ty);
        if fx.cx.is_backend_immediate(layout) {
            // These sorts of types are immediates that we can store
            // in an Value without an alloca.
        } else if fx.cx.is_backend_scalar_pair(layout) {
            // We allow pairs and uses of any of their 2 fields.
        } else {
            // These sorts of types require an alloca. Note that
            // is_llvm_immediate() may *still* be true, particularly
            // for newtypes, but we currently force some types
            // (e.g., structs) into an alloca unconditionally, just so
            // that we don't have to deal with having two pathways
            // (gep vs extractvalue etc).
            analyzer.not_ssa(mir::Local::new(index));
        }
    }

    analyzer.non_ssa_locals
}

struct LocalAnalyzer<'mir, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    fx: &'mir FunctionCx<'a, 'tcx, Bx>,
    dominators: Dominators<mir::BasicBlock>,
    non_ssa_locals: BitSet<mir::Local>,
    // The location of the first visited direct assignment to each
    // local, or an invalid location (out of bounds `block` index).
    first_assignment: IndexVec<mir::Local, Location>,
}

impl<Bx: BuilderMethods<'a, 'tcx>> LocalAnalyzer<'mir, 'a, 'tcx, Bx> {
    fn new(fx: &'mir FunctionCx<'a, 'tcx, Bx>) -> Self {
        let invalid_location =
            mir::BasicBlock::new(fx.mir.basic_blocks().len()).start_location();
        let mut analyzer = LocalAnalyzer {
            fx,
            dominators: fx.mir.dominators(),
            non_ssa_locals: BitSet::new_empty(fx.mir.local_decls.len()),
            first_assignment: IndexVec::from_elem(invalid_location, &fx.mir.local_decls)
        };

        // Arguments get assigned to by means of the function being called
        for arg in fx.mir.args_iter() {
            analyzer.first_assignment[arg] = mir::START_BLOCK.start_location();
        }

        analyzer
    }

    fn first_assignment(&self, local: mir::Local) -> Option<Location> {
        let location = self.first_assignment[local];
        if location.block.index() < self.fx.mir.basic_blocks().len() {
            Some(location)
        } else {
            None
        }
    }

    fn not_ssa(&mut self, local: mir::Local) {
        debug!("marking {:?} as non-SSA", local);
        self.non_ssa_locals.insert(local);
    }

    fn assign(&mut self, local: mir::Local, location: Location) {
        if self.first_assignment(local).is_some() {
            self.not_ssa(local);
        } else {
            self.first_assignment[local] = location;
        }
    }
}

impl<'mir, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> Visitor<'tcx>
    for LocalAnalyzer<'mir, 'a, 'tcx, Bx>
{
    fn visit_assign(&mut self,
                    place: &mir::Place<'tcx>,
                    rvalue: &mir::Rvalue<'tcx>,
                    location: Location) {
        debug!("visit_assign(place={:?}, rvalue={:?})", place, rvalue);

        if let mir::Place::Base(mir::PlaceBase::Local(index)) = *place {
            self.assign(index, location);
            if !self.fx.rvalue_creates_operand(rvalue) {
                self.not_ssa(index);
            }
        } else {
            self.visit_place(
                place,
                PlaceContext::MutatingUse(MutatingUseContext::Store),
                location
            );
        }

        self.visit_rvalue(rvalue, location);
    }

    fn visit_terminator_kind(&mut self,
                             kind: &mir::TerminatorKind<'tcx>,
                             location: Location) {
        let check = match *kind {
            mir::TerminatorKind::Call {
                func: mir::Operand::Constant(ref c),
                ref args, ..
            } => match c.ty.sty {
                ty::FnDef(did, _) => Some((did, args)),
                _ => None,
            },
            _ => None,
        };
        if let Some((def_id, args)) = check {
            if Some(def_id) == self.fx.cx.tcx().lang_items().box_free_fn() {
                // box_free(x) shares with `drop x` the property that it
                // is not guaranteed to be statically dominated by the
                // definition of x, so x must always be in an alloca.
                if let mir::Operand::Move(ref place) = args[0] {
                    self.visit_place(
                        place,
                        PlaceContext::MutatingUse(MutatingUseContext::Drop),
                        location
                    );
                }
            }
        }

        self.super_terminator_kind(kind, location);
    }

    fn visit_place(&mut self,
                   place: &mir::Place<'tcx>,
                   context: PlaceContext,
                   location: Location) {
        debug!("visit_place(place={:?}, context={:?})", place, context);
        let cx = self.fx.cx;

        if let mir::Place::Projection(ref proj) = *place {
            // Allow uses of projections that are ZSTs or from scalar fields.
            let is_consume = match context {
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) |
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => true,
                _ => false
            };
            if is_consume {
                let base_ty = proj.base.ty(self.fx.mir, cx.tcx());
                let base_ty = self.fx.monomorphize(&base_ty);

                // ZSTs don't require any actual memory access.
                let elem_ty = base_ty
                    .projection_ty(cx.tcx(), &proj.elem)
                    .ty;
                let elem_ty = self.fx.monomorphize(&elem_ty);
                if cx.layout_of(elem_ty).is_zst() {
                    return;
                }

                if let mir::ProjectionElem::Field(..) = proj.elem {
                    let layout = cx.layout_of(base_ty.ty);
                    if cx.is_backend_immediate(layout) || cx.is_backend_scalar_pair(layout) {
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
                return self.visit_place(
                    &proj.base,
                    PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy),
                    location
                );
            }
        }

        self.super_place(place, context, location);
    }

    fn visit_local(&mut self,
                   &local: &mir::Local,
                   context: PlaceContext,
                   location: Location) {
        match context {
            PlaceContext::MutatingUse(MutatingUseContext::Call) => {
                self.assign(local, location);
            }

            PlaceContext::NonUse(_) |
            PlaceContext::MutatingUse(MutatingUseContext::Retag) => {}

            PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) |
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => {
                // Reads from uninitialized variables (e.g., in dead code, after
                // optimizations) require locals to be in (uninitialized) memory.
                // N.B., there can be uninitialized reads of a local visited after
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

            PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect) |
            PlaceContext::MutatingUse(MutatingUseContext::Store) |
            PlaceContext::MutatingUse(MutatingUseContext::AsmOutput) |
            PlaceContext::MutatingUse(MutatingUseContext::Borrow) |
            PlaceContext::MutatingUse(MutatingUseContext::Projection) |
            PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow) |
            PlaceContext::NonMutatingUse(NonMutatingUseContext::UniqueBorrow) |
            PlaceContext::NonMutatingUse(NonMutatingUseContext::ShallowBorrow) |
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => {
                self.not_ssa(local);
            }

            PlaceContext::MutatingUse(MutatingUseContext::Drop) => {
                let ty = self.fx.mir.local_decls[local].ty;
                let ty = self.fx.monomorphize(&ty);

                // Only need the place if we're actually dropping it.
                if self.fx.cx.type_needs_drop(ty) {
                    self.not_ssa(local);
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CleanupKind {
    NotCleanup,
    Funclet,
    Internal { funclet: mir::BasicBlock }
}

impl CleanupKind {
    pub fn funclet_bb(self, for_bb: mir::BasicBlock) -> Option<mir::BasicBlock> {
        match self {
            CleanupKind::NotCleanup => None,
            CleanupKind::Funclet => Some(for_bb),
            CleanupKind::Internal { funclet } => Some(funclet),
        }
    }
}

pub fn cleanup_kinds(mir: &mir::Body<'_>) -> IndexVec<mir::BasicBlock, CleanupKind> {
    fn discover_masters<'tcx>(result: &mut IndexVec<mir::BasicBlock, CleanupKind>,
                              mir: &mir::Body<'tcx>) {
        for (bb, data) in mir.basic_blocks().iter_enumerated() {
            match data.terminator().kind {
                TerminatorKind::Goto { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Abort |
                TerminatorKind::Return |
                TerminatorKind::GeneratorDrop |
                TerminatorKind::Unreachable |
                TerminatorKind::SwitchInt { .. } |
                TerminatorKind::Yield { .. } |
                TerminatorKind::FalseEdges { .. } |
                TerminatorKind::FalseUnwind { .. } => {
                    /* nothing to do */
                }
                TerminatorKind::Call { cleanup: unwind, .. } |
                TerminatorKind::Assert { cleanup: unwind, .. } |
                TerminatorKind::DropAndReplace { unwind, .. } |
                TerminatorKind::Drop { unwind, .. } => {
                    if let Some(unwind) = unwind {
                        debug!("cleanup_kinds: {:?}/{:?} registering {:?} as funclet",
                               bb, data, unwind);
                        result[unwind] = CleanupKind::Funclet;
                    }
                }
            }
        }
    }

    fn propagate<'tcx>(result: &mut IndexVec<mir::BasicBlock, CleanupKind>,
                       mir: &mir::Body<'tcx>) {
        let mut funclet_succs = IndexVec::from_elem(None, mir.basic_blocks());

        let mut set_successor = |funclet: mir::BasicBlock, succ| {
            match funclet_succs[funclet] {
                ref mut s @ None => {
                    debug!("set_successor: updating successor of {:?} to {:?}",
                           funclet, succ);
                    *s = Some(succ);
                },
                Some(s) => if s != succ {
                    span_bug!(mir.span, "funclet {:?} has 2 parents - {:?} and {:?}",
                              funclet, s, succ);
                }
            }
        };

        for (bb, data) in traversal::reverse_postorder(mir) {
            let funclet = match result[bb] {
                CleanupKind::NotCleanup => continue,
                CleanupKind::Funclet => bb,
                CleanupKind::Internal { funclet } => funclet,
            };

            debug!("cleanup_kinds: {:?}/{:?}/{:?} propagating funclet {:?}",
                   bb, data, result[bb], funclet);

            for &succ in data.terminator().successors() {
                let kind = result[succ];
                debug!("cleanup_kinds: propagating {:?} to {:?}/{:?}",
                       funclet, succ, kind);
                match kind {
                    CleanupKind::NotCleanup => {
                        result[succ] = CleanupKind::Internal { funclet };
                    }
                    CleanupKind::Funclet => {
                        if funclet != succ {
                            set_successor(funclet, succ);
                        }
                    }
                    CleanupKind::Internal { funclet: succ_funclet } => {
                        if funclet != succ_funclet {
                            // `succ` has 2 different funclet going into it, so it must
                            // be a funclet by itself.

                            debug!("promoting {:?} to a funclet and updating {:?}", succ,
                                   succ_funclet);
                            result[succ] = CleanupKind::Funclet;
                            set_successor(succ_funclet, succ);
                            set_successor(funclet, succ);
                        }
                    }
                }
            }
        }
    }

    let mut result = IndexVec::from_elem(CleanupKind::NotCleanup, mir.basic_blocks());

    discover_masters(&mut result, mir);
    propagate(&mut result, mir);
    debug!("cleanup_kinds: result={:?}", result);
    result
}
