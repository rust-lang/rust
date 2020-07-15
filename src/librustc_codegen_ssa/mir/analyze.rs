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
use rustc_middle::ty;
use rustc_middle::ty::layout::{HasTyCtxt, TyAndLayout};
use rustc_target::abi::LayoutOf;

pub fn non_ssa_locals<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    fx: &FunctionCx<'a, 'tcx, Bx>,
) -> BitSet<mir::Local> {
    trace!("non_ssa_locals({:?})", fx.instance.def_id());

    let mir = fx.mir;
    let mut analyzer = LocalAnalyzer::new(fx);

    for (block, data) in traversal::reverse_postorder(mir) {
        analyzer.visit_basic_block_data(block, data);
    }

    for (local, decl) in mir.local_decls.iter_enumerated() {
        let ty = fx.monomorphize(&decl.ty);
        debug!("local {:?} has type `{}`", local, ty);
        let layout = fx.cx.spanned_layout_of(ty, decl.source_info.span);

        if ty_requires_alloca(&analyzer.fx, layout) {
            analyzer.not_ssa(local);
        }
    }

    analyzer.non_ssa_locals
}

struct LocalAnalyzer<'mir, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    fx: &'mir FunctionCx<'a, 'tcx, Bx>,
    dominators: Dominators<mir::BasicBlock>,
    non_ssa_locals: BitSet<mir::Local>,

    /// The location of the first visited direct assignment to each local.
    first_assignment: IndexVec<mir::Local, Option<Location>>,
}

impl<Bx: BuilderMethods<'a, 'tcx>> LocalAnalyzer<'mir, 'a, 'tcx, Bx> {
    fn new(fx: &'mir FunctionCx<'a, 'tcx, Bx>) -> Self {
        let dominators = fx.mir.dominators();
        let mut analyzer = LocalAnalyzer {
            fx,
            dominators,
            non_ssa_locals: BitSet::new_empty(fx.mir.local_decls.len()),
            first_assignment: IndexVec::from_elem(None, &fx.mir.local_decls),
        };

        // Arguments get assigned to by means of the function being called
        for arg in fx.mir.args_iter() {
            analyzer.assign(arg, mir::START_BLOCK.start_location());
        }

        analyzer
    }

    fn not_ssa(&mut self, local: mir::Local) {
        debug!("marking {:?} as non-SSA", local);
        self.non_ssa_locals.insert(local);
    }

    fn assign(&mut self, local: mir::Local, location: Location) {
        if self.first_assignment[local].is_some() {
            self.not_ssa(local);
        } else {
            self.first_assignment[local] = Some(location);
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
                match c.literal.ty.kind {
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

    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        mut context: PlaceContext,
        location: Location,
    ) {
        debug!("visit_place(place={:?}, context={:?})", place, context);

        // Non-uses do not force locals onto the stack.
        if !context.is_use() {
            return;
        }

        let mir::Place { local, projection } = *place;

        // Reads from ZSTs do not require memory accesses and do not count when determining what
        // needs to live on the stack.
        if is_consume(context) {
            let ty = place.ty(self.fx.mir, self.fx.cx.tcx()).ty;
            let ty = self.fx.monomorphize(&ty);
            let span = self.fx.mir.local_decls[local].source_info.span;
            if self.fx.cx.spanned_layout_of(ty, span).is_zst() {
                return;
            }
        }

        let mut has_disqualifying_projection = false;

        let mut projection: &[_] = projection.as_ref();
        while let [ref proj_base @ .., elem] = *projection {
            projection = proj_base;
            self.super_projection_elem(local, proj_base, elem, context, location);

            // Projections like `(*x)[12]` are allowed but not `*(x[12])`, since a `Deref` of a
            // local acts like a `Copy` of that local.
            if let mir::PlaceElem::Deref = elem {
                has_disqualifying_projection = false;
                context = PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy);
                continue;
            }

            // Ignoring `Deref`s, the only allowed projections are reads of scalar fields.
            if is_consume(context) && matches!(elem, mir::ProjectionElem::Field(..)) {
                let base_ty = mir::Place::ty_from(local, proj_base, self.fx.mir, self.fx.cx.tcx());
                let base_ty = self.fx.monomorphize(&base_ty);
                let span = self.fx.mir.local_decls[local].source_info.span;
                let layout = self.fx.cx.spanned_layout_of(base_ty.ty, span);

                if !ty_requires_alloca(self.fx, layout) {
                    continue;
                }
            }

            has_disqualifying_projection = true;
        }

        if has_disqualifying_projection {
            self.not_ssa(local);
        }

        self.visit_local(&local, context, location);
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
                let ssa_read = match self.first_assignment[local] {
                    Some(assignment_location) => {
                        assignment_location.dominates(location, &self.dominators)
                    }
                    None => false,
                };
                if !ssa_read {
                    self.not_ssa(local);
                }
            }

            PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => {
                unreachable!("We always use the original context from `visit_place`")
            }

            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Borrow
                | MutatingUseContext::AddressOf,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Inspect
                | NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::UniqueBorrow
                | NonMutatingUseContext::ShallowBorrow
                | NonMutatingUseContext::AddressOf,
            ) => {
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

/// Returns `true` if locals of this type need to be allocated on the stack.
fn ty_requires_alloca<'a, 'tcx>(
    fx: &FunctionCx<'a, 'tcx, impl BuilderMethods<'a, 'tcx>>,
    ty: TyAndLayout<'tcx>,
) -> bool {
    // Currently, this returns `true` for ADTs that are otherwise small enough to qualify. For
    // example, `struct Newtype(i32)`. This way, every type has a single way to extract data
    // (gep, extractvalue, etc.).
    !fx.cx.is_backend_immediate(ty) && !fx.cx.is_backend_scalar_pair(ty)
}

fn is_consume(context: PlaceContext) -> bool {
    matches!(
        context,
        PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy | NonMutatingUseContext::Move)
    )
}
