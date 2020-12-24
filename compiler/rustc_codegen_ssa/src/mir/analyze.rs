//! An analysis to determine which locals require allocas and
//! which do not.

use super::FunctionCx;
use crate::traits::*;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::mir::traversal;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::{self, Location, TerminatorKind};
use rustc_middle::ty;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_target::abi::LayoutOf;

pub fn non_ssa_locals<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    fx: &FunctionCx<'a, 'tcx, Bx>,
) -> BitSet<mir::Local> {
    let mir = fx.mir;
    let mut analyzer = LocalAnalyzer::new(fx);

    analyzer.visit_body(&mir);

    for (local, decl) in mir.local_decls.iter_enumerated() {
        let ty = fx.monomorphize(decl.ty);
        debug!("local {:?} has type `{}`", local, ty);
        let layout = fx.cx.spanned_layout_of(ty, decl.source_info.span);
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
            analyzer.not_ssa(local);
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
        let invalid_location = mir::BasicBlock::new(fx.mir.basic_blocks().len()).start_location();
        let dominators = fx.mir.dominators();
        let mut analyzer = LocalAnalyzer {
            fx,
            dominators,
            non_ssa_locals: BitSet::new_empty(fx.mir.local_decls.len()),
            first_assignment: IndexVec::from_elem(invalid_location, &fx.mir.local_decls),
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

    fn process_place(
        &mut self,
        place_ref: &mir::PlaceRef<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        let cx = self.fx.cx;

        if let &[ref proj_base @ .., elem] = place_ref.projection {
            let mut base_context = if context.is_mutating_use() {
                PlaceContext::MutatingUse(MutatingUseContext::Projection)
            } else {
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection)
            };

            // Allow uses of projections that are ZSTs or from scalar fields.
            let is_consume = matches!(
                context,
                PlaceContext::NonMutatingUse(
                    NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
                )
            );
            if is_consume {
                let base_ty =
                    mir::Place::ty_from(place_ref.local, proj_base, self.fx.mir, cx.tcx());
                let base_ty = self.fx.monomorphize(base_ty);

                // ZSTs don't require any actual memory access.
                let elem_ty = base_ty.projection_ty(cx.tcx(), self.fx.monomorphize(elem)).ty;
                let span = self.fx.mir.local_decls[place_ref.local].source_info.span;
                if cx.spanned_layout_of(elem_ty, span).is_zst() {
                    return;
                }

                if let mir::ProjectionElem::Field(..) = elem {
                    let layout = cx.spanned_layout_of(base_ty.ty, span);
                    if cx.is_backend_immediate(layout) || cx.is_backend_scalar_pair(layout) {
                        // Recurse with the same context, instead of `Projection`,
                        // potentially stopping at non-operand projections,
                        // which would trigger `not_ssa` on locals.
                        base_context = context;
                    }
                }
            }

            if let mir::ProjectionElem::Deref = elem {
                // Deref projections typically only read the pointer.
                // (the exception being `VarDebugInfo` contexts, handled below)
                base_context = PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy);

                // Indirect debuginfo requires going through memory, that only
                // the debugger accesses, following our emitted DWARF pointer ops.
                //
                // FIXME(eddyb) Investigate the possibility of relaxing this, but
                // note that `llvm.dbg.declare` *must* be used for indirect places,
                // even if we start using `llvm.dbg.value` for all other cases,
                // as we don't necessarily know when the value changes, but only
                // where it lives in memory.
                //
                // It's possible `llvm.dbg.declare` could support starting from
                // a pointer that doesn't point to an `alloca`, but this would
                // only be useful if we know the pointer being `Deref`'d comes
                // from an immutable place, and if `llvm.dbg.declare` calls
                // must be at the very start of the function, then only function
                // arguments could contain such pointers.
                if context == PlaceContext::NonUse(NonUseContext::VarDebugInfo) {
                    // We use `NonUseContext::VarDebugInfo` for the base,
                    // which might not force the base local to memory,
                    // so we have to do it manually.
                    self.visit_local(&place_ref.local, context, location);
                }
            }

            // `NonUseContext::VarDebugInfo` needs to flow all the
            // way down to the base local (see `visit_local`).
            if context == PlaceContext::NonUse(NonUseContext::VarDebugInfo) {
                base_context = context;
            }

            self.process_place(
                &mir::PlaceRef { local: place_ref.local, projection: proj_base },
                base_context,
                location,
            );
            // HACK(eddyb) this emulates the old `visit_projection_elem`, this
            // entire `visit_place`-like `process_place` method should be rewritten,
            // now that we have moved to the "slice of projections" representation.
            if let mir::ProjectionElem::Index(local) = elem {
                self.visit_local(
                    &local,
                    PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy),
                    location,
                );
            }
        } else {
            // FIXME this is super_place code, is repeated here to avoid cloning place or changing
            // visit_place API
            let mut context = context;

            if !place_ref.projection.is_empty() {
                context = if context.is_mutating_use() {
                    PlaceContext::MutatingUse(MutatingUseContext::Projection)
                } else {
                    PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection)
                };
            }

            self.visit_local(&place_ref.local, context, location);
            self.visit_projection(place_ref.local, place_ref.projection, context, location);
        }
    }
}

impl<'mir, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> Visitor<'tcx>
    for LocalAnalyzer<'mir, 'a, 'tcx, Bx>
{
    fn visit_assign(
        &mut self,
        place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        debug!("visit_assign(place={:?}, rvalue={:?})", place, rvalue);

        if let Some(index) = place.as_local() {
            self.assign(index, location);
            let decl_span = self.fx.mir.local_decls[index].source_info.span;
            if !self.fx.rvalue_creates_operand(rvalue, decl_span) {
                self.not_ssa(index);
            }
        } else {
            self.visit_place(place, PlaceContext::MutatingUse(MutatingUseContext::Store), location);
        }

        self.visit_rvalue(rvalue, location);
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        let check = match terminator.kind {
            mir::TerminatorKind::Call { func: mir::Operand::Constant(ref c), ref args, .. } => {
                match *c.literal.ty.kind() {
                    ty::FnDef(did, _) => Some((did, args)),
                    _ => None,
                }
            }
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
                        location,
                    );
                }
            }
        }

        self.super_terminator(terminator, location);
    }

    fn visit_place(&mut self, place: &mir::Place<'tcx>, context: PlaceContext, location: Location) {
        debug!("visit_place(place={:?}, context={:?})", place, context);
        self.process_place(&place.as_ref(), context, location);
    }

    fn visit_local(&mut self, &local: &mir::Local, context: PlaceContext, location: Location) {
        match context {
            PlaceContext::MutatingUse(MutatingUseContext::Call)
            | PlaceContext::MutatingUse(MutatingUseContext::Yield) => {
                self.assign(local, location);
            }

            PlaceContext::NonUse(_) | PlaceContext::MutatingUse(MutatingUseContext::Retag) => {}

            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
            ) => {
                // Reads from uninitialized variables (e.g., in dead code, after
                // optimizations) require locals to be in (uninitialized) memory.
                // N.B., there can be uninitialized reads of a local visited after
                // an assignment to that local, if they happen on disjoint paths.
                let ssa_read = match self.first_assignment(local) {
                    Some(assignment_location) => {
                        assignment_location.dominates(location, &self.dominators)
                    }
                    None => false,
                };
                if !ssa_read {
                    self.not_ssa(local);
                }
            }

            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Borrow
                | MutatingUseContext::AddressOf
                | MutatingUseContext::Projection,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Inspect
                | NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::UniqueBorrow
                | NonMutatingUseContext::ShallowBorrow
                | NonMutatingUseContext::AddressOf
                | NonMutatingUseContext::Projection,
            ) => {
                self.not_ssa(local);
            }

            PlaceContext::MutatingUse(MutatingUseContext::Drop) => {
                let ty = self.fx.mir.local_decls[local].ty;
                let ty = self.fx.monomorphize(ty);

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
    Internal { funclet: mir::BasicBlock },
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
    fn discover_masters<'tcx>(
        result: &mut IndexVec<mir::BasicBlock, CleanupKind>,
        mir: &mir::Body<'tcx>,
    ) {
        for (bb, data) in mir.basic_blocks().iter_enumerated() {
            match data.terminator().kind {
                TerminatorKind::Goto { .. }
                | TerminatorKind::Resume
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::GeneratorDrop
                | TerminatorKind::Unreachable
                | TerminatorKind::SwitchInt { .. }
                | TerminatorKind::Yield { .. }
                | TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. }
                | TerminatorKind::InlineAsm { .. } => { /* nothing to do */ }
                TerminatorKind::Call { cleanup: unwind, .. }
                | TerminatorKind::Assert { cleanup: unwind, .. }
                | TerminatorKind::DropAndReplace { unwind, .. }
                | TerminatorKind::Drop { unwind, .. } => {
                    if let Some(unwind) = unwind {
                        debug!(
                            "cleanup_kinds: {:?}/{:?} registering {:?} as funclet",
                            bb, data, unwind
                        );
                        result[unwind] = CleanupKind::Funclet;
                    }
                }
            }
        }
    }

    fn propagate<'tcx>(result: &mut IndexVec<mir::BasicBlock, CleanupKind>, mir: &mir::Body<'tcx>) {
        let mut funclet_succs = IndexVec::from_elem(None, mir.basic_blocks());

        let mut set_successor = |funclet: mir::BasicBlock, succ| match funclet_succs[funclet] {
            ref mut s @ None => {
                debug!("set_successor: updating successor of {:?} to {:?}", funclet, succ);
                *s = Some(succ);
            }
            Some(s) => {
                if s != succ {
                    span_bug!(
                        mir.span,
                        "funclet {:?} has 2 parents - {:?} and {:?}",
                        funclet,
                        s,
                        succ
                    );
                }
            }
        };

        for (bb, data) in traversal::reverse_postorder(mir) {
            let funclet = match result[bb] {
                CleanupKind::NotCleanup => continue,
                CleanupKind::Funclet => bb,
                CleanupKind::Internal { funclet } => funclet,
            };

            debug!(
                "cleanup_kinds: {:?}/{:?}/{:?} propagating funclet {:?}",
                bb, data, result[bb], funclet
            );

            for &succ in data.terminator().successors() {
                let kind = result[succ];
                debug!("cleanup_kinds: propagating {:?} to {:?}/{:?}", funclet, succ, kind);
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

                            debug!(
                                "promoting {:?} to a funclet and updating {:?}",
                                succ, succ_funclet
                            );
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
