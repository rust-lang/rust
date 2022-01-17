//! An analysis to determine which locals require allocas and
//! which do not.

use super::FunctionCx;
use crate::traits::*;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::traversal;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Location, TerminatorKind};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf};

pub fn non_ssa_locals<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    fx: &FunctionCx<'a, 'tcx, Bx>,
) -> BitSet<mir::Local> {
    let mir = fx.mir;
    let dominators = mir.dominators();
    let locals = mir
        .local_decls
        .iter()
        .map(|decl| {
            let ty = fx.monomorphize(decl.ty);
            let layout = fx.cx.spanned_layout_of(ty, decl.source_info.span);
            if layout.is_zst() {
                LocalKind::ZST
            } else if fx.cx.is_backend_immediate(layout) || fx.cx.is_backend_scalar_pair(layout) {
                LocalKind::Unused
            } else {
                LocalKind::Memory
            }
        })
        .collect();

    let mut analyzer = LocalAnalyzer { fx, dominators, locals };

    // Arguments get assigned to by means of the function being called
    for arg in mir.args_iter() {
        analyzer.assign(arg, mir::START_BLOCK.start_location());
    }

    // If there exists a local definition that dominates all uses of that local,
    // the definition should be visited first. Traverse blocks in preorder which
    // is a topological sort of dominance partial order.
    for (bb, data) in traversal::preorder(&mir) {
        analyzer.visit_basic_block_data(bb, data);
    }

    let mut non_ssa_locals = BitSet::new_empty(analyzer.locals.len());
    for (local, kind) in analyzer.locals.iter_enumerated() {
        if matches!(kind, LocalKind::Memory) {
            non_ssa_locals.insert(local);
        }
    }

    non_ssa_locals
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum LocalKind {
    ZST,
    /// A local that requires an alloca.
    Memory,
    /// A scalar or a scalar pair local that is neither defined nor used.
    Unused,
    /// A scalar or a scalar pair local with a single definition that dominates all uses.
    SSA(mir::Location),
}

struct LocalAnalyzer<'mir, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    fx: &'mir FunctionCx<'a, 'tcx, Bx>,
    dominators: Dominators<mir::BasicBlock>,
    locals: IndexVec<mir::Local, LocalKind>,
}

impl<'mir, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> LocalAnalyzer<'mir, 'a, 'tcx, Bx> {
    fn assign(&mut self, local: mir::Local, location: Location) {
        let kind = &mut self.locals[local];
        match *kind {
            LocalKind::ZST => {}
            LocalKind::Memory => {}
            LocalKind::Unused => {
                *kind = LocalKind::SSA(location);
            }
            LocalKind::SSA(_) => {
                *kind = LocalKind::Memory;
            }
        }
    }

    fn process_place(
        &mut self,
        place_ref: &mir::PlaceRef<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        let cx = self.fx.cx;

        if let Some((place_base, elem)) = place_ref.last_projection() {
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
                let base_ty = place_base.ty(self.fx.mir, cx.tcx());
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
                base_context = PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy);
            }

            self.process_place(&place_base, base_context, location);
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
            self.visit_local(&place_ref.local, context, location);
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

        if let Some(local) = place.as_local() {
            self.assign(local, location);
            if self.locals[local] != LocalKind::Memory {
                let decl_span = self.fx.mir.local_decls[local].source_info.span;
                if !self.fx.rvalue_creates_operand(rvalue, decl_span) {
                    self.locals[local] = LocalKind::Memory;
                }
            }
        } else {
            self.visit_place(place, PlaceContext::MutatingUse(MutatingUseContext::Store), location);
        }

        self.visit_rvalue(rvalue, location);
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
            ) => match &mut self.locals[local] {
                LocalKind::ZST => {}
                LocalKind::Memory => {}
                LocalKind::SSA(def) if def.dominates(location, &self.dominators) => {}
                // Reads from uninitialized variables (e.g., in dead code, after
                // optimizations) require locals to be in (uninitialized) memory.
                // N.B., there can be uninitialized reads of a local visited after
                // an assignment to that local, if they happen on disjoint paths.
                kind @ (LocalKind::Unused | LocalKind::SSA(_)) => {
                    *kind = LocalKind::Memory;
                }
            },

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
                self.locals[local] = LocalKind::Memory;
            }

            PlaceContext::MutatingUse(MutatingUseContext::Drop) => {
                let kind = &mut self.locals[local];
                if *kind != LocalKind::Memory {
                    let ty = self.fx.mir.local_decls[local].ty;
                    let ty = self.fx.monomorphize(ty);
                    if self.fx.cx.type_needs_drop(ty) {
                        // Only need the place if we're actually dropping it.
                        *kind = LocalKind::Memory;
                    }
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
                | TerminatorKind::FalseUnwind { .. } => { /* nothing to do */ }
                TerminatorKind::Call { cleanup: unwind, .. }
                | TerminatorKind::InlineAsm { cleanup: unwind, .. }
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
