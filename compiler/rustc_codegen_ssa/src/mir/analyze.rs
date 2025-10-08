//! An analysis to determine which locals require allocas and
//! which do not.

use rustc_abi as abi;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, DefLocation, Location, TerminatorKind, traversal};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::{bug, span_bug};
use tracing::debug;

use super::FunctionCx;
use crate::traits::*;

pub(crate) fn non_ssa_locals<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    fx: &FunctionCx<'a, 'tcx, Bx>,
    traversal_order: &[mir::BasicBlock],
) -> DenseBitSet<mir::Local> {
    let mir = fx.mir;
    let dominators = mir.basic_blocks.dominators();
    let locals = mir
        .local_decls
        .iter()
        .map(|decl| {
            let ty = fx.monomorphize(decl.ty);
            let layout = fx.cx.spanned_layout_of(ty, decl.source_info.span);
            if layout.is_zst() { LocalKind::ZST } else { LocalKind::Unused }
        })
        .collect();

    let mut analyzer = LocalAnalyzer { fx, dominators, locals };

    // Arguments get assigned to by means of the function being called
    for arg in mir.args_iter() {
        analyzer.define(arg, DefLocation::Argument);
    }

    // If there exists a local definition that dominates all uses of that local,
    // the definition should be visited first. Traverse blocks in an order that
    // is a topological sort of dominance partial order.
    for bb in traversal_order.iter().copied() {
        let data = &mir.basic_blocks[bb];
        analyzer.visit_basic_block_data(bb, data);
    }

    let mut non_ssa_locals = DenseBitSet::new_empty(analyzer.locals.len());
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
    SSA(DefLocation),
}

struct LocalAnalyzer<'a, 'b, 'tcx, Bx: BuilderMethods<'b, 'tcx>> {
    fx: &'a FunctionCx<'b, 'tcx, Bx>,
    dominators: &'a Dominators<mir::BasicBlock>,
    locals: IndexVec<mir::Local, LocalKind>,
}

impl<'a, 'b, 'tcx, Bx: BuilderMethods<'b, 'tcx>> LocalAnalyzer<'a, 'b, 'tcx, Bx> {
    fn define(&mut self, local: mir::Local, location: DefLocation) {
        let fx = self.fx;
        let kind = &mut self.locals[local];
        let decl = &fx.mir.local_decls[local];
        match *kind {
            LocalKind::ZST => {}
            LocalKind::Memory => {}
            LocalKind::Unused => {
                let ty = fx.monomorphize(decl.ty);
                let layout = fx.cx.spanned_layout_of(ty, decl.source_info.span);
                *kind =
                    if fx.cx.is_backend_immediate(layout) || fx.cx.is_backend_scalar_pair(layout) {
                        LocalKind::SSA(location)
                    } else {
                        LocalKind::Memory
                    };
            }
            LocalKind::SSA(_) => *kind = LocalKind::Memory,
        }
    }

    fn process_place(
        &mut self,
        place_ref: &mir::PlaceRef<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        if !place_ref.projection.is_empty() {
            const COPY_CONTEXT: PlaceContext =
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy);

            // `PlaceElem::Index` is the only variant that can mention other `Local`s,
            // so check for those up-front before any potential short-circuits.
            for elem in place_ref.projection {
                if let mir::PlaceElem::Index(index_local) = *elem {
                    self.visit_local(index_local, COPY_CONTEXT, location);
                }
            }

            // If our local is already memory, nothing can make it *more* memory
            // so we don't need to bother checking the projections further.
            if self.locals[place_ref.local] == LocalKind::Memory {
                return;
            }

            if place_ref.is_indirect_first_projection() {
                // If this starts with a `Deref`, we only need to record a read of the
                // pointer being dereferenced, as all the subsequent projections are
                // working on a place which is always supported. (And because we're
                // looking at codegen MIR, it can only happen as the first projection.)
                self.visit_local(place_ref.local, COPY_CONTEXT, location);
                return;
            }

            if context.is_mutating_use() {
                // If it's a mutating use it doesn't matter what the projections are,
                // if there are *any* then we need a place to write. (For example,
                // `_1 = Foo()` works in SSA but `_2.0 = Foo()` does not.)
                let mut_projection = PlaceContext::MutatingUse(MutatingUseContext::Projection);
                self.visit_local(place_ref.local, mut_projection, location);
                return;
            }

            // Scan through to ensure the only projections are those which
            // `FunctionCx::maybe_codegen_consume_direct` can handle.
            let base_ty = self.fx.monomorphized_place_ty(mir::PlaceRef::from(place_ref.local));
            let mut layout = self.fx.cx.layout_of(base_ty);
            for elem in place_ref.projection {
                layout = match *elem {
                    mir::PlaceElem::Field(fidx, ..) => layout.field(self.fx.cx, fidx.as_usize()),
                    mir::PlaceElem::Downcast(_, vidx)
                        if let abi::Variants::Single { index: single_variant } =
                            layout.variants
                            && vidx == single_variant =>
                    {
                        layout.for_variant(self.fx.cx, vidx)
                    }
                    _ => {
                        self.locals[place_ref.local] = LocalKind::Memory;
                        return;
                    }
                }
            }
            debug_assert!(
                !self.fx.cx.is_backend_ref(layout),
                "Post-projection {place_ref:?} layout should be non-Ref, but it's {layout:?}",
            );
        }

        // Even with supported projections, we still need to have `visit_local`
        // check for things that can't be done in SSA (like `SharedBorrow`).
        self.visit_local(place_ref.local, context, location);
    }
}

impl<'a, 'b, 'tcx, Bx: BuilderMethods<'b, 'tcx>> Visitor<'tcx> for LocalAnalyzer<'a, 'b, 'tcx, Bx> {
    fn visit_assign(
        &mut self,
        place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        debug!("visit_assign(place={:?}, rvalue={:?})", place, rvalue);

        if let Some(local) = place.as_local() {
            self.define(local, DefLocation::Assignment(location));
        } else {
            self.visit_place(place, PlaceContext::MutatingUse(MutatingUseContext::Store), location);
        }

        self.visit_rvalue(rvalue, location);
    }

    fn visit_place(&mut self, place: &mir::Place<'tcx>, context: PlaceContext, location: Location) {
        debug!("visit_place(place={:?}, context={:?})", place, context);
        self.process_place(&place.as_ref(), context, location);
    }

    fn visit_local(&mut self, local: mir::Local, context: PlaceContext, location: Location) {
        match context {
            PlaceContext::MutatingUse(MutatingUseContext::Call) => {
                let call = location.block;
                let TerminatorKind::Call { target, .. } =
                    self.fx.mir.basic_blocks[call].terminator().kind
                else {
                    bug!()
                };
                self.define(local, DefLocation::CallReturn { call, target });
            }

            PlaceContext::NonUse(_)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::PlaceMention)
            | PlaceContext::MutatingUse(MutatingUseContext::Retag) => {}

            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy
                | NonMutatingUseContext::Move
                // Inspect covers things like `PtrMetadata` and `Discriminant`
                // which we can treat similar to `Copy` use for the purpose of
                // whether we can use SSA variables for things.
                | NonMutatingUseContext::Inspect,
            ) => match &mut self.locals[local] {
                LocalKind::ZST => {}
                LocalKind::Memory => {}
                LocalKind::SSA(def) if def.dominates(location, self.dominators) => {}
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
                | MutatingUseContext::Deinit
                | MutatingUseContext::SetDiscriminant
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Borrow
                | MutatingUseContext::RawBorrow
                | MutatingUseContext::Projection,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::FakeBorrow
                | NonMutatingUseContext::RawBorrow
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

            PlaceContext::MutatingUse(MutatingUseContext::Yield) => bug!(),
        }
    }

    fn visit_statement_debuginfo(&mut self, _: &mir::StmtDebugInfo<'tcx>, _: Location) {
        // Debuginfo does not generate actual code.
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CleanupKind {
    NotCleanup,
    Funclet,
    Internal { funclet: mir::BasicBlock },
}

impl CleanupKind {
    pub(crate) fn funclet_bb(self, for_bb: mir::BasicBlock) -> Option<mir::BasicBlock> {
        match self {
            CleanupKind::NotCleanup => None,
            CleanupKind::Funclet => Some(for_bb),
            CleanupKind::Internal { funclet } => Some(funclet),
        }
    }
}

/// MSVC requires unwinding code to be split to a tree of *funclets*, where each funclet can only
/// branch to itself or to its parent. Luckily, the code we generates matches this pattern.
/// Recover that structure in an analyze pass.
pub(crate) fn cleanup_kinds(mir: &mir::Body<'_>) -> IndexVec<mir::BasicBlock, CleanupKind> {
    fn discover_masters<'tcx>(
        result: &mut IndexSlice<mir::BasicBlock, CleanupKind>,
        mir: &mir::Body<'tcx>,
    ) {
        for (bb, data) in mir.basic_blocks.iter_enumerated() {
            match data.terminator().kind {
                TerminatorKind::Goto { .. }
                | TerminatorKind::UnwindResume
                | TerminatorKind::UnwindTerminate(_)
                | TerminatorKind::Return
                | TerminatorKind::TailCall { .. }
                | TerminatorKind::CoroutineDrop
                | TerminatorKind::Unreachable
                | TerminatorKind::SwitchInt { .. }
                | TerminatorKind::Yield { .. }
                | TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. } => { /* nothing to do */ }
                TerminatorKind::Call { unwind, .. }
                | TerminatorKind::InlineAsm { unwind, .. }
                | TerminatorKind::Assert { unwind, .. }
                | TerminatorKind::Drop { unwind, .. } => {
                    if let mir::UnwindAction::Cleanup(unwind) = unwind {
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

    fn propagate<'tcx>(
        result: &mut IndexSlice<mir::BasicBlock, CleanupKind>,
        mir: &mir::Body<'tcx>,
    ) {
        let mut funclet_succs = IndexVec::from_elem(None, &mir.basic_blocks);

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

            for succ in data.terminator().successors() {
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

    let mut result = IndexVec::from_elem(CleanupKind::NotCleanup, &mir.basic_blocks);

    discover_masters(&mut result, mir);
    propagate(&mut result, mir);
    debug!("cleanup_kinds: result={:?}", result);
    result
}
