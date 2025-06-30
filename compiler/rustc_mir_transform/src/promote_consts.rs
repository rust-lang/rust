//! A pass that promotes borrows of constant rvalues.
//!
//! The rvalues considered constant are trees of temps, each with exactly one
//! initialization, and holding a constant value with no interior mutability.
//! They are placed into a new MIR constant body in `promoted` and the borrow
//! rvalue is replaced with a `Literal::Promoted` using the index into
//! `promoted` of that constant MIR.
//!
//! This pass assumes that every use is dominated by an initialization and can
//! otherwise silence errors, if move analysis runs after promotion on broken
//! MIR.

use std::assert_matches::assert_matches;
use std::cell::Cell;
use std::{cmp, iter, mem};

use either::{Left, Right};
use rustc_const_eval::check_consts::{ConstCx, qualifs};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::visit::{MutVisitor, MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, GenericArgs, List, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, mir, span_bug};
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use tracing::{debug, instrument};

/// A `MirPass` for promotion.
///
/// Promotion is the extraction of promotable temps into separate MIR bodies so they can have
/// `'static` lifetime.
///
/// After this pass is run, `promoted_fragments` will hold the MIR body corresponding to each
/// newly created `Constant`.
#[derive(Default)]
pub(super) struct PromoteTemps<'tcx> {
    // Must use `Cell` because `run_pass` takes `&self`, not `&mut self`.
    pub promoted_fragments: Cell<IndexVec<Promoted, Body<'tcx>>>,
}

impl<'tcx> crate::MirPass<'tcx> for PromoteTemps<'tcx> {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // There's not really any point in promoting errorful MIR.
        //
        // This does not include MIR that failed const-checking, which we still try to promote.
        if let Err(_) = body.return_ty().error_reported() {
            debug!("PromoteTemps: MIR had errors");
            return;
        }
        if body.source.promoted.is_some() {
            return;
        }

        let ccx = ConstCx::new(tcx, body);
        let (mut temps, all_candidates) = collect_temps_and_candidates(&ccx);

        let promotable_candidates = validate_candidates(&ccx, &mut temps, all_candidates);

        let promoted = promote_candidates(body, tcx, temps, promotable_candidates);
        self.promoted_fragments.set(promoted);
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// State of a temporary during collection and promotion.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum TempState {
    /// No references to this temp.
    Undefined,
    /// One direct assignment and any number of direct uses.
    /// A borrow of this temp is promotable if the assigned
    /// value is qualified as constant.
    Defined { location: Location, uses: usize, valid: Result<(), ()> },
    /// Any other combination of assignments/uses.
    Unpromotable,
    /// This temp was part of an rvalue which got extracted
    /// during promotion and needs cleanup.
    PromotedOut,
}

/// A "root candidate" for promotion, which will become the
/// returned value in a promoted MIR, unless it's a subset
/// of a larger candidate.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Candidate {
    location: Location,
}

struct Collector<'a, 'tcx> {
    ccx: &'a ConstCx<'a, 'tcx>,
    temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
}

impl<'tcx> Visitor<'tcx> for Collector<'_, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn visit_local(&mut self, index: Local, context: PlaceContext, location: Location) {
        // We're only interested in temporaries and the return place
        match self.ccx.body.local_kind(index) {
            LocalKind::Arg => return,
            LocalKind::Temp if self.ccx.body.local_decls[index].is_user_variable() => return,
            LocalKind::ReturnPointer | LocalKind::Temp => {}
        }

        // Ignore drops, if the temp gets promoted,
        // then it's constant and thus drop is noop.
        // Non-uses are also irrelevant.
        if context.is_drop() || !context.is_use() {
            debug!(is_drop = context.is_drop(), is_use = context.is_use());
            return;
        }

        let temp = &mut self.temps[index];
        debug!(?temp);
        *temp = match *temp {
            TempState::Undefined => match context {
                PlaceContext::MutatingUse(MutatingUseContext::Store | MutatingUseContext::Call) => {
                    TempState::Defined { location, uses: 0, valid: Err(()) }
                }
                _ => TempState::Unpromotable,
            },
            TempState::Defined { ref mut uses, .. } => {
                // We always allow borrows, even mutable ones, as we need
                // to promote mutable borrows of some ZSTs e.g., `&mut []`.
                let allowed_use = match context {
                    PlaceContext::MutatingUse(MutatingUseContext::Borrow)
                    | PlaceContext::NonMutatingUse(_) => true,
                    PlaceContext::MutatingUse(_) | PlaceContext::NonUse(_) => false,
                };
                debug!(?allowed_use);
                if allowed_use {
                    *uses += 1;
                    return;
                }
                TempState::Unpromotable
            }
            TempState::Unpromotable | TempState::PromotedOut => TempState::Unpromotable,
        };
        debug!(?temp);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        if let Rvalue::Ref(..) = *rvalue {
            self.candidates.push(Candidate { location });
        }
    }
}

fn collect_temps_and_candidates<'tcx>(
    ccx: &ConstCx<'_, 'tcx>,
) -> (IndexVec<Local, TempState>, Vec<Candidate>) {
    let mut collector = Collector {
        temps: IndexVec::from_elem(TempState::Undefined, &ccx.body.local_decls),
        candidates: vec![],
        ccx,
    };
    for (bb, data) in traversal::reverse_postorder(ccx.body) {
        collector.visit_basic_block_data(bb, data);
    }
    (collector.temps, collector.candidates)
}

/// Checks whether locals that appear in a promotion context (`Candidate`) are actually promotable.
///
/// This wraps an `Item`, and has access to all fields of that `Item` via `Deref` coercion.
struct Validator<'a, 'tcx> {
    ccx: &'a ConstCx<'a, 'tcx>,
    temps: &'a mut IndexSlice<Local, TempState>,
    /// For backwards compatibility, we are promoting function calls in `const`/`static`
    /// initializers. But we want to avoid evaluating code that might panic and that otherwise would
    /// not have been evaluated, so we only promote such calls in basic blocks that are guaranteed
    /// to execute. In other words, we only promote such calls in basic blocks that are definitely
    /// not dead code. Here we cache the result of computing that set of basic blocks.
    promotion_safe_blocks: Option<FxHashSet<BasicBlock>>,
}

impl<'a, 'tcx> std::ops::Deref for Validator<'a, 'tcx> {
    type Target = ConstCx<'a, 'tcx>;

    fn deref(&self) -> &Self::Target {
        self.ccx
    }
}

struct Unpromotable;

impl<'tcx> Validator<'_, 'tcx> {
    fn validate_candidate(&mut self, candidate: Candidate) -> Result<(), Unpromotable> {
        let Left(statement) = self.body.stmt_at(candidate.location) else { bug!() };
        let Some((_, Rvalue::Ref(_, kind, place))) = statement.kind.as_assign() else { bug!() };

        // We can only promote interior borrows of promotable temps (non-temps
        // don't get promoted anyway).
        self.validate_local(place.local)?;

        // The reference operation itself must be promotable.
        // (Needs to come after `validate_local` to avoid ICEs.)
        self.validate_ref(*kind, place)?;

        // We do not check all the projections (they do not get promoted anyway),
        // but we do stay away from promoting anything involving a dereference.
        if place.projection.contains(&ProjectionElem::Deref) {
            return Err(Unpromotable);
        }

        Ok(())
    }

    // FIXME(eddyb) maybe cache this?
    fn qualif_local<Q: qualifs::Qualif>(&mut self, local: Local) -> bool {
        let TempState::Defined { location: loc, .. } = self.temps[local] else {
            return false;
        };

        let stmt_or_term = self.body.stmt_at(loc);
        match stmt_or_term {
            Left(statement) => {
                let Some((_, rhs)) = statement.kind.as_assign() else {
                    span_bug!(statement.source_info.span, "{:?} is not an assignment", statement)
                };
                qualifs::in_rvalue::<Q, _>(self.ccx, &mut |l| self.qualif_local::<Q>(l), rhs)
            }
            Right(terminator) => {
                assert_matches!(terminator.kind, TerminatorKind::Call { .. });
                let return_ty = self.body.local_decls[local].ty;
                Q::in_any_value_of_ty(self.ccx, return_ty)
            }
        }
    }

    fn validate_local(&mut self, local: Local) -> Result<(), Unpromotable> {
        let TempState::Defined { location: loc, uses, valid } = self.temps[local] else {
            return Err(Unpromotable);
        };

        // We cannot promote things that need dropping, since the promoted value would not get
        // dropped.
        if self.qualif_local::<qualifs::NeedsDrop>(local) {
            return Err(Unpromotable);
        }

        if valid.is_ok() {
            return Ok(());
        }

        let ok = {
            let stmt_or_term = self.body.stmt_at(loc);
            match stmt_or_term {
                Left(statement) => {
                    let Some((_, rhs)) = statement.kind.as_assign() else {
                        span_bug!(
                            statement.source_info.span,
                            "{:?} is not an assignment",
                            statement
                        )
                    };
                    self.validate_rvalue(rhs)
                }
                Right(terminator) => match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => {
                        self.validate_call(func, args, loc.block)
                    }
                    TerminatorKind::Yield { .. } => Err(Unpromotable),
                    kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                },
            }
        };

        self.temps[local] = match ok {
            Ok(()) => TempState::Defined { location: loc, uses, valid: Ok(()) },
            Err(_) => TempState::Unpromotable,
        };

        ok
    }

    fn validate_place(&mut self, place: PlaceRef<'tcx>) -> Result<(), Unpromotable> {
        let Some((place_base, elem)) = place.last_projection() else {
            return self.validate_local(place.local);
        };

        // Validate topmost projection, then recurse.
        match elem {
            // Recurse directly.
            ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subtype(_)
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::UnwrapUnsafeBinder(_) => {}

            // Never recurse.
            ProjectionElem::OpaqueCast(..) | ProjectionElem::Downcast(..) => {
                return Err(Unpromotable);
            }

            ProjectionElem::Deref => {
                // When a static is used by-value, that gets desugared to `*STATIC_ADDR`,
                // and we need to be able to promote this. So check if this deref matches
                // that specific pattern.

                // We need to make sure this is a `Deref` of a local with no further projections.
                // Discussion can be found at
                // https://github.com/rust-lang/rust/pull/74945#discussion_r463063247
                if let Some(local) = place_base.as_local()
                    && let TempState::Defined { location, .. } = self.temps[local]
                    && let Left(def_stmt) = self.body.stmt_at(location)
                    && let Some((_, Rvalue::Use(Operand::Constant(c)))) = def_stmt.kind.as_assign()
                    && let Some(did) = c.check_static_ptr(self.tcx)
                    // Evaluating a promoted may not read statics except if it got
                    // promoted from a static (this is a CTFE check). So we
                    // can only promote static accesses inside statics.
                    && let Some(hir::ConstContext::Static(..)) = self.const_kind
                    && !self.tcx.is_thread_local_static(did)
                {
                    // Recurse.
                } else {
                    return Err(Unpromotable);
                }
            }
            ProjectionElem::Index(local) => {
                // Only accept if we can predict the index and are indexing an array.
                if let TempState::Defined { location: loc, .. } = self.temps[local]
                    && let Left(statement) =  self.body.stmt_at(loc)
                    && let Some((_, Rvalue::Use(Operand::Constant(c)))) = statement.kind.as_assign()
                    && let Some(idx) = c.const_.try_eval_target_usize(self.tcx, self.typing_env)
                    // Determine the type of the thing we are indexing.
                    && let ty::Array(_, len) = place_base.ty(self.body, self.tcx).ty.kind()
                    // It's an array; determine its length.
                    && let Some(len) = len.try_to_target_usize(self.tcx)
                    // If the index is in-bounds, go ahead.
                    && idx < len
                {
                    self.validate_local(local)?;
                    // Recurse.
                } else {
                    return Err(Unpromotable);
                }
            }

            ProjectionElem::Field(..) => {
                let base_ty = place_base.ty(self.body, self.tcx).ty;
                if base_ty.is_union() {
                    // No promotion of union field accesses.
                    return Err(Unpromotable);
                }
            }
        }

        self.validate_place(place_base)
    }

    fn validate_operand(&mut self, operand: &Operand<'tcx>) -> Result<(), Unpromotable> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.validate_place(place.as_ref()),

            // The qualifs for a constant (e.g. `HasMutInterior`) are checked in
            // `validate_rvalue` upon access.
            Operand::Constant(c) => {
                if let Some(def_id) = c.check_static_ptr(self.tcx) {
                    // Only allow statics (not consts) to refer to other statics.
                    // FIXME(eddyb) does this matter at all for promotion?
                    // FIXME(RalfJung) it makes little sense to not promote this in `fn`/`const fn`,
                    // and in `const` this cannot occur anyway. The only concern is that we might
                    // promote even `let x = &STATIC` which would be useless, but this applies to
                    // promotion inside statics as well.
                    let is_static = matches!(self.const_kind, Some(hir::ConstContext::Static(_)));
                    if !is_static {
                        return Err(Unpromotable);
                    }

                    let is_thread_local = self.tcx.is_thread_local_static(def_id);
                    if is_thread_local {
                        return Err(Unpromotable);
                    }
                }

                Ok(())
            }
        }
    }

    fn validate_ref(&mut self, kind: BorrowKind, place: &Place<'tcx>) -> Result<(), Unpromotable> {
        match kind {
            // Reject these borrow types just to be safe.
            // FIXME(RalfJung): could we allow them? Should we? No point in it until we have a
            // usecase.
            BorrowKind::Fake(_) | BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture } => {
                return Err(Unpromotable);
            }

            BorrowKind::Shared => {
                let has_mut_interior = self.qualif_local::<qualifs::HasMutInterior>(place.local);
                if has_mut_interior {
                    return Err(Unpromotable);
                }
            }

            // FIXME: consider changing this to only promote &mut [] for default borrows,
            // also forbidding two phase borrows
            BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow } => {
                let ty = place.ty(self.body, self.tcx).ty;

                // In theory, any zero-sized value could be borrowed
                // mutably without consequences. However, only &mut []
                // is allowed right now.
                if let ty::Array(_, len) = ty.kind() {
                    match len.try_to_target_usize(self.tcx) {
                        Some(0) => {}
                        _ => return Err(Unpromotable),
                    }
                } else {
                    return Err(Unpromotable);
                }
            }
        }

        Ok(())
    }

    fn validate_rvalue(&mut self, rvalue: &Rvalue<'tcx>) -> Result<(), Unpromotable> {
        match rvalue {
            Rvalue::Use(operand)
            | Rvalue::Repeat(operand, _)
            | Rvalue::WrapUnsafeBinder(operand, _) => {
                self.validate_operand(operand)?;
            }
            Rvalue::CopyForDeref(place) => {
                let op = &Operand::Copy(*place);
                self.validate_operand(op)?
            }

            Rvalue::Discriminant(place) | Rvalue::Len(place) => {
                self.validate_place(place.as_ref())?
            }

            Rvalue::ThreadLocalRef(_) => return Err(Unpromotable),

            // ptr-to-int casts are not possible in consts and thus not promotable
            Rvalue::Cast(CastKind::PointerExposeProvenance, _, _) => return Err(Unpromotable),

            // all other casts including int-to-ptr casts are fine, they just use the integer value
            // at pointer type.
            Rvalue::Cast(_, operand, _) => {
                self.validate_operand(operand)?;
            }

            Rvalue::NullaryOp(op, _) => match op {
                NullOp::SizeOf => {}
                NullOp::AlignOf => {}
                NullOp::OffsetOf(_) => {}
                NullOp::UbChecks => {}
                NullOp::ContractChecks => {}
            },

            Rvalue::ShallowInitBox(_, _) => return Err(Unpromotable),

            Rvalue::UnaryOp(op, operand) => {
                match op {
                    // These operations can never fail.
                    UnOp::Neg | UnOp::Not | UnOp::PtrMetadata => {}
                }

                self.validate_operand(operand)?;
            }

            Rvalue::BinaryOp(op, box (lhs, rhs)) => {
                let op = *op;
                let lhs_ty = lhs.ty(self.body, self.tcx);

                if let ty::RawPtr(_, _) | ty::FnPtr(..) = lhs_ty.kind() {
                    // Raw and fn pointer operations are not allowed inside consts and thus not
                    // promotable.
                    assert_matches!(
                        op,
                        BinOp::Eq
                            | BinOp::Ne
                            | BinOp::Le
                            | BinOp::Lt
                            | BinOp::Ge
                            | BinOp::Gt
                            | BinOp::Offset
                    );
                    return Err(Unpromotable);
                }

                match op {
                    BinOp::Div | BinOp::Rem => {
                        if lhs_ty.is_integral() {
                            let sz = lhs_ty.primitive_size(self.tcx);
                            // Integer division: the RHS must be a non-zero const.
                            let rhs_val = match rhs {
                                Operand::Constant(c) => {
                                    c.const_.try_eval_scalar_int(self.tcx, self.typing_env)
                                }
                                _ => None,
                            };
                            match rhs_val.map(|x| x.to_uint(sz)) {
                                // for the zero test, int vs uint does not matter
                                Some(x) if x != 0 => {}        // okay
                                _ => return Err(Unpromotable), // value not known or 0 -- not okay
                            }
                            // Furthermore, for signed division, we also have to exclude `int::MIN /
                            // -1`.
                            if lhs_ty.is_signed() {
                                match rhs_val.map(|x| x.to_int(sz)) {
                                    Some(-1) | None => {
                                        // The RHS is -1 or unknown, so we have to be careful.
                                        // But is the LHS int::MIN?
                                        let lhs_val = match lhs {
                                            Operand::Constant(c) => c
                                                .const_
                                                .try_eval_scalar_int(self.tcx, self.typing_env),
                                            _ => None,
                                        };
                                        let lhs_min = sz.signed_int_min();
                                        match lhs_val.map(|x| x.to_int(sz)) {
                                            // okay
                                            Some(x) if x != lhs_min => {}

                                            // value not known or int::MIN -- not okay
                                            _ => return Err(Unpromotable),
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    // The remaining operations can never fail.
                    BinOp::Eq
                    | BinOp::Ne
                    | BinOp::Le
                    | BinOp::Lt
                    | BinOp::Ge
                    | BinOp::Gt
                    | BinOp::Cmp
                    | BinOp::Offset
                    | BinOp::Add
                    | BinOp::AddUnchecked
                    | BinOp::AddWithOverflow
                    | BinOp::Sub
                    | BinOp::SubUnchecked
                    | BinOp::SubWithOverflow
                    | BinOp::Mul
                    | BinOp::MulUnchecked
                    | BinOp::MulWithOverflow
                    | BinOp::BitXor
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::Shl
                    | BinOp::ShlUnchecked
                    | BinOp::Shr
                    | BinOp::ShrUnchecked => {}
                }

                self.validate_operand(lhs)?;
                self.validate_operand(rhs)?;
            }

            Rvalue::RawPtr(_, place) => {
                // We accept `&raw *`, i.e., raw reborrows -- creating a raw pointer is
                // no problem, only using it is.
                if let Some((place_base, ProjectionElem::Deref)) = place.as_ref().last_projection()
                {
                    let base_ty = place_base.ty(self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind() {
                        return self.validate_place(place_base);
                    }
                }
                return Err(Unpromotable);
            }

            Rvalue::Ref(_, kind, place) => {
                // Special-case reborrows to be more like a copy of the reference.
                let mut place_simplified = place.as_ref();
                if let Some((place_base, ProjectionElem::Deref)) =
                    place_simplified.last_projection()
                {
                    let base_ty = place_base.ty(self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind() {
                        place_simplified = place_base;
                    }
                }

                self.validate_place(place_simplified)?;

                // Check that the reference is fine (using the original place!).
                // (Needs to come after `validate_place` to avoid ICEs.)
                self.validate_ref(*kind, place)?;
            }

            Rvalue::Aggregate(_, operands) => {
                for o in operands {
                    self.validate_operand(o)?;
                }
            }
        }

        Ok(())
    }

    /// Computes the sets of blocks of this MIR that are definitely going to be executed
    /// if the function returns successfully. That makes it safe to promote calls in them
    /// that might fail.
    fn promotion_safe_blocks(body: &mir::Body<'tcx>) -> FxHashSet<BasicBlock> {
        let mut safe_blocks = FxHashSet::default();
        let mut safe_block = START_BLOCK;
        loop {
            safe_blocks.insert(safe_block);
            // Let's see if we can find another safe block.
            safe_block = match body.basic_blocks[safe_block].terminator().kind {
                TerminatorKind::Goto { target } => target,
                TerminatorKind::Call { target: Some(target), .. }
                | TerminatorKind::Drop { target, .. } => {
                    // This calls a function or the destructor. `target` does not get executed if
                    // the callee loops or panics. But in both cases the const already fails to
                    // evaluate, so we are fine considering `target` a safe block for promotion.
                    target
                }
                TerminatorKind::Assert { target, .. } => {
                    // Similar to above, we only consider successful execution.
                    target
                }
                _ => {
                    // No next safe block.
                    break;
                }
            };
        }
        safe_blocks
    }

    /// Returns whether the block is "safe" for promotion, which means it cannot be dead code.
    /// We use this to avoid promoting operations that can fail in dead code.
    fn is_promotion_safe_block(&mut self, block: BasicBlock) -> bool {
        let body = self.body;
        let safe_blocks =
            self.promotion_safe_blocks.get_or_insert_with(|| Self::promotion_safe_blocks(body));
        safe_blocks.contains(&block)
    }

    fn validate_call(
        &mut self,
        callee: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        block: BasicBlock,
    ) -> Result<(), Unpromotable> {
        // Validate the operands. If they fail, there's no question -- we cannot promote.
        self.validate_operand(callee)?;
        for arg in args {
            self.validate_operand(&arg.node)?;
        }

        // Functions marked `#[rustc_promotable]` are explicitly allowed to be promoted, so we can
        // accept them at this point.
        let fn_ty = callee.ty(self.body, self.tcx);
        if let ty::FnDef(def_id, _) = *fn_ty.kind() {
            if self.tcx.is_promotable_const_fn(def_id) {
                return Ok(());
            }
        }

        // Ideally, we'd stop here and reject the rest.
        // But for backward compatibility, we have to accept some promotion in const/static
        // initializers. Inline consts are explicitly excluded, they are more recent so we have no
        // backwards compatibility reason to allow more promotion inside of them.
        let promote_all_fn = matches!(
            self.const_kind,
            Some(hir::ConstContext::Static(_) | hir::ConstContext::Const { inline: false })
        );
        if !promote_all_fn {
            return Err(Unpromotable);
        }
        // Make sure the callee is a `const fn`.
        let is_const_fn = match *fn_ty.kind() {
            ty::FnDef(def_id, _) => self.tcx.is_const_fn(def_id),
            _ => false,
        };
        if !is_const_fn {
            return Err(Unpromotable);
        }
        // The problem is, this may promote calls to functions that panic.
        // We don't want to introduce compilation errors if there's a panic in a call in dead code.
        // So we ensure that this is not dead code.
        if !self.is_promotion_safe_block(block) {
            return Err(Unpromotable);
        }
        // This passed all checks, so let's accept.
        Ok(())
    }
}

fn validate_candidates(
    ccx: &ConstCx<'_, '_>,
    temps: &mut IndexSlice<Local, TempState>,
    mut candidates: Vec<Candidate>,
) -> Vec<Candidate> {
    let mut validator = Validator { ccx, temps, promotion_safe_blocks: None };

    candidates.retain(|&candidate| validator.validate_candidate(candidate).is_ok());
    candidates
}

struct Promoter<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source: &'a mut Body<'tcx>,
    promoted: Body<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,
    extra_statements: &'a mut Vec<(Location, Statement<'tcx>)>,

    /// Used to assemble the required_consts list while building the promoted.
    required_consts: Vec<ConstOperand<'tcx>>,

    /// If true, all nested temps are also kept in the
    /// source MIR, not moved to the promoted MIR.
    keep_original: bool,

    /// If true, add the new const (the promoted) to the required_consts of the parent MIR.
    /// This is initially false and then set by the visitor when it encounters a `Call` terminator.
    add_to_required: bool,
}

impl<'a, 'tcx> Promoter<'a, 'tcx> {
    fn new_block(&mut self) -> BasicBlock {
        let span = self.promoted.span;
        self.promoted.basic_blocks_mut().push(BasicBlockData::new(
            Some(Terminator {
                source_info: SourceInfo::outermost(span),
                kind: TerminatorKind::Return,
            }),
            false,
        ))
    }

    fn assign(&mut self, dest: Local, rvalue: Rvalue<'tcx>, span: Span) {
        let last = self.promoted.basic_blocks.last_index().unwrap();
        let data = &mut self.promoted[last];
        data.statements.push(Statement::new(
            SourceInfo::outermost(span),
            StatementKind::Assign(Box::new((Place::from(dest), rvalue))),
        ));
    }

    fn is_temp_kind(&self, local: Local) -> bool {
        self.source.local_kind(local) == LocalKind::Temp
    }

    /// Copies the initialization of this temp to the
    /// promoted MIR, recursing through temps.
    fn promote_temp(&mut self, temp: Local) -> Local {
        let old_keep_original = self.keep_original;
        let loc = match self.temps[temp] {
            TempState::Defined { location, uses, .. } if uses > 0 => {
                if uses > 1 {
                    self.keep_original = true;
                }
                location
            }
            state => {
                span_bug!(self.promoted.span, "{:?} not promotable: {:?}", temp, state);
            }
        };
        if !self.keep_original {
            self.temps[temp] = TempState::PromotedOut;
        }

        let num_stmts = self.source[loc.block].statements.len();
        let new_temp = self.promoted.local_decls.push(LocalDecl::new(
            self.source.local_decls[temp].ty,
            self.source.local_decls[temp].source_info.span,
        ));

        debug!("promote({:?} @ {:?}/{:?}, {:?})", temp, loc, num_stmts, self.keep_original);

        // First, take the Rvalue or Call out of the source MIR,
        // or duplicate it, depending on keep_original.
        if loc.statement_index < num_stmts {
            let (mut rvalue, source_info) = {
                let statement = &mut self.source[loc.block].statements[loc.statement_index];
                let StatementKind::Assign(box (_, rhs)) = &mut statement.kind else {
                    span_bug!(statement.source_info.span, "{:?} is not an assignment", statement);
                };

                (
                    if self.keep_original {
                        rhs.clone()
                    } else {
                        let unit = Rvalue::Use(Operand::Constant(Box::new(ConstOperand {
                            span: statement.source_info.span,
                            user_ty: None,
                            const_: Const::zero_sized(self.tcx.types.unit),
                        })));
                        mem::replace(rhs, unit)
                    },
                    statement.source_info,
                )
            };

            self.visit_rvalue(&mut rvalue, loc);
            self.assign(new_temp, rvalue, source_info.span);
        } else {
            let terminator = if self.keep_original {
                self.source[loc.block].terminator().clone()
            } else {
                let terminator = self.source[loc.block].terminator_mut();
                let target = match &terminator.kind {
                    TerminatorKind::Call { target: Some(target), .. } => *target,
                    kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                };
                Terminator {
                    source_info: terminator.source_info,
                    kind: mem::replace(&mut terminator.kind, TerminatorKind::Goto { target }),
                }
            };

            match terminator.kind {
                TerminatorKind::Call {
                    mut func, mut args, call_source: desugar, fn_span, ..
                } => {
                    // This promoted involves a function call, so it may fail to evaluate. Let's
                    // make sure it is added to `required_consts` so that failure cannot get lost.
                    self.add_to_required = true;

                    self.visit_operand(&mut func, loc);
                    for arg in &mut args {
                        self.visit_operand(&mut arg.node, loc);
                    }

                    let last = self.promoted.basic_blocks.last_index().unwrap();
                    let new_target = self.new_block();

                    *self.promoted[last].terminator_mut() = Terminator {
                        kind: TerminatorKind::Call {
                            func,
                            args,
                            unwind: UnwindAction::Continue,
                            destination: Place::from(new_temp),
                            target: Some(new_target),
                            call_source: desugar,
                            fn_span,
                        },
                        source_info: SourceInfo::outermost(terminator.source_info.span),
                        ..terminator
                    };
                }
                kind => {
                    span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                }
            };
        };

        self.keep_original = old_keep_original;
        new_temp
    }

    fn promote_candidate(
        mut self,
        candidate: Candidate,
        next_promoted_index: Promoted,
    ) -> Body<'tcx> {
        let def = self.source.source.def_id();
        let (mut rvalue, promoted_op) = {
            let promoted = &mut self.promoted;
            let tcx = self.tcx;
            let mut promoted_operand = |ty, span| {
                promoted.span = span;
                promoted.local_decls[RETURN_PLACE] = LocalDecl::new(ty, span);
                let args = tcx.erase_regions(GenericArgs::identity_for_item(tcx, def));
                let uneval =
                    mir::UnevaluatedConst { def, args, promoted: Some(next_promoted_index) };

                ConstOperand { span, user_ty: None, const_: Const::Unevaluated(uneval, ty) }
            };

            let blocks = self.source.basic_blocks.as_mut();
            let local_decls = &mut self.source.local_decls;
            let loc = candidate.location;
            let statement = &mut blocks[loc.block].statements[loc.statement_index];
            let StatementKind::Assign(box (_, Rvalue::Ref(region, borrow_kind, place))) =
                &mut statement.kind
            else {
                bug!()
            };

            // Use the underlying local for this (necessarily interior) borrow.
            debug_assert!(region.is_erased());
            let ty = local_decls[place.local].ty;
            let span = statement.source_info.span;

            let ref_ty =
                Ty::new_ref(tcx, tcx.lifetimes.re_erased, ty, borrow_kind.to_mutbl_lossy());

            let mut projection = vec![PlaceElem::Deref];
            projection.extend(place.projection);
            place.projection = tcx.mk_place_elems(&projection);

            // Create a temp to hold the promoted reference.
            // This is because `*r` requires `r` to be a local,
            // otherwise we would use the `promoted` directly.
            let mut promoted_ref = LocalDecl::new(ref_ty, span);
            promoted_ref.source_info = statement.source_info;
            let promoted_ref = local_decls.push(promoted_ref);
            assert_eq!(self.temps.push(TempState::Unpromotable), promoted_ref);

            let promoted_operand = promoted_operand(ref_ty, span);
            let promoted_ref_statement = Statement::new(
                statement.source_info,
                StatementKind::Assign(Box::new((
                    Place::from(promoted_ref),
                    Rvalue::Use(Operand::Constant(Box::new(promoted_operand))),
                ))),
            );
            self.extra_statements.push((loc, promoted_ref_statement));

            (
                Rvalue::Ref(
                    tcx.lifetimes.re_erased,
                    *borrow_kind,
                    Place {
                        local: mem::replace(&mut place.local, promoted_ref),
                        projection: List::empty(),
                    },
                ),
                promoted_operand,
            )
        };

        assert_eq!(self.new_block(), START_BLOCK);
        self.visit_rvalue(
            &mut rvalue,
            Location { block: START_BLOCK, statement_index: usize::MAX },
        );

        let span = self.promoted.span;
        self.assign(RETURN_PLACE, rvalue, span);

        // Now that we did promotion, we know whether we'll want to add this to `required_consts` of
        // the surrounding MIR body.
        if self.add_to_required {
            self.source.required_consts.as_mut().unwrap().push(promoted_op);
        }

        self.promoted.set_required_consts(self.required_consts);

        self.promoted
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        if self.is_temp_kind(*local) {
            *local = self.promote_temp(*local);
        }
    }

    fn visit_const_operand(&mut self, constant: &mut ConstOperand<'tcx>, _location: Location) {
        if constant.const_.is_required_const() {
            self.required_consts.push(*constant);
        }

        // Skipping `super_constant` as the visitor is otherwise only looking for locals.
    }
}

fn promote_candidates<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    mut temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
) -> IndexVec<Promoted, Body<'tcx>> {
    // Visit candidates in reverse, in case they're nested.
    debug!(promote_candidates = ?candidates);

    // eagerly fail fast
    if candidates.is_empty() {
        return IndexVec::new();
    }

    let mut promotions = IndexVec::new();

    let mut extra_statements = vec![];
    for candidate in candidates.into_iter().rev() {
        let Location { block, statement_index } = candidate.location;
        if let StatementKind::Assign(box (place, _)) = &body[block].statements[statement_index].kind
        {
            if let Some(local) = place.as_local() {
                if temps[local] == TempState::PromotedOut {
                    // Already promoted.
                    continue;
                }
            }
        }

        // Declare return place local so that `mir::Body::new` doesn't complain.
        let initial_locals = iter::once(LocalDecl::new(tcx.types.never, body.span)).collect();

        let mut scope = body.source_scopes[body.source_info(candidate.location).scope].clone();
        scope.parent_scope = None;

        let mut promoted = Body::new(
            body.source, // `promoted` gets filled in below
            IndexVec::new(),
            IndexVec::from_elem_n(scope, 1),
            initial_locals,
            IndexVec::new(),
            0,
            vec![],
            body.span,
            None,
            body.tainted_by_errors,
        );
        promoted.phase = MirPhase::Analysis(AnalysisPhase::Initial);

        let promoter = Promoter {
            promoted,
            tcx,
            source: body,
            temps: &mut temps,
            extra_statements: &mut extra_statements,
            keep_original: false,
            add_to_required: false,
            required_consts: Vec::new(),
        };

        let mut promoted = promoter.promote_candidate(candidate, promotions.next_index());
        promoted.source.promoted = Some(promotions.next_index());
        promotions.push(promoted);
    }

    // Insert each of `extra_statements` before its indicated location, which
    // has to be done in reverse location order, to not invalidate the rest.
    extra_statements.sort_by_key(|&(loc, _)| cmp::Reverse(loc));
    for (loc, statement) in extra_statements {
        body[loc.block].statements.insert(loc.statement_index, statement);
    }

    // Eliminate assignments to, and drops of promoted temps.
    let promoted = |index: Local| temps[index] == TempState::PromotedOut;
    for block in body.basic_blocks_mut() {
        block.statements.retain(|statement| match &statement.kind {
            StatementKind::Assign(box (place, _)) => {
                if let Some(index) = place.as_local() {
                    !promoted(index)
                } else {
                    true
                }
            }
            StatementKind::StorageLive(index) | StatementKind::StorageDead(index) => {
                !promoted(*index)
            }
            _ => true,
        });
        let terminator = block.terminator_mut();
        if let TerminatorKind::Drop { place, target, .. } = &terminator.kind {
            if let Some(index) = place.as_local() {
                if promoted(index) {
                    terminator.kind = TerminatorKind::Goto { target: *target };
                }
            }
        }
    }

    promotions
}
