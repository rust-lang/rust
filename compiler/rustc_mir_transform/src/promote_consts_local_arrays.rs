//! A pass that promotes const local arrays to rodata.
//!
//! The rvalues considered constant are trees of temps,
//! each with exactly one initialization, and holding
//! a constant value with no interior mutability.
//! They are placed into a new MIR constant body in
//! `promoted` and the borrow rvalue is replaced with
//! a `Literal::Promoted` using the index into `promoted`
//! of that constant MIR.
//!
//! This pass assumes that every use is dominated by an
//! initialization and can otherwise silence errors, if
//! move analysis runs after promotion on broken MIR.

use either::{Left, Right};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::mir;
use rustc_middle::mir::visit::{
    MutVisitor, MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::*;
use rustc_middle::ty::GenericArgs;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_span::Span;

use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_span::source_map::Spanned;

use std::assert_matches::assert_matches;
use std::cell::Cell;
use std::{cmp, mem};

use rustc_const_eval::check_consts::{qualifs, ConstCx};

/// A `MirPass` for promotion.
///
/// Promotion is the extraction of promotable temps into separate MIR bodies so they can have
/// `'static` lifetime.
///
/// After this pass is run, `promoted_fragments` will hold the MIR body corresponding to each
/// newly created `Constant`.
// FIXME: consider merging to `PromoteTemps` pass.
#[derive(Default)]
pub struct PromoteArraysOpt<'tcx> {
    pub promoted_fragments: Cell<IndexVec<Promoted, Body<'tcx>>>,
}

// LLVM optimizes the load of 16 byte as a single `mov`.
// Bigger values make more `mov` instructions generated.
// While changing code as this lint suggests, it becomes
// a single load (`lea`) of an address in `.rodata`.
const STACK_THRESHOLD: u64 = 16;

impl<'tcx> MirPass<'tcx> for PromoteArraysOpt<'tcx> {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("running on body: {:?}", body.source.def_id());
        // There's not really any point in promoting errorful MIR.
        //
        // This does not include MIR that failed const-checking, which we still try to promote.
        if let Err(_) = body.return_ty().error_reported() {
            debug!("MIR had errors");
            return;
        }
        if body.source.promoted.is_some() {
            return;
        }

        // Ignore static/const items. Already processed by PromoteTemps
        if let Some(ctx) = tcx.hir().body_const_context(body.source.def_id()) {
            use hir::ConstContext::*;
            match ctx {
                Static(_) | Const { inline: _ } => return,
                _ => {}
            }
        }

        let ccx = ConstCx::new(tcx, body);
        let (mut temps, all_candidates, already_promoted) = collect_temps_and_candidates(&ccx);

        let promotable_candidates = validate_candidates(&ccx, &mut temps, &all_candidates);
        debug!(candidates = ?promotable_candidates);

        let promoted =
            promote_candidates(body, tcx, temps, promotable_candidates, already_promoted);
        self.promoted_fragments.set(promoted);
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
    already_promoted: usize,
}

impl<'tcx> Visitor<'tcx> for Collector<'_, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn visit_local(&mut self, index: Local, context: PlaceContext, location: Location) {
        // We're only interested in temporaries
        match self.ccx.body.local_kind(index) {
            LocalKind::Arg | LocalKind::ReturnPointer => return,
            LocalKind::Temp => {}
        }

        {
            let is_user_variable = self.ccx.body.local_decls[index].is_user_variable();
            debug!(?is_user_variable);
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
                use NonMutatingUseContext as Ctxt;
                // We only allow non-mutating use of arrays.
                let allowed_use = match context {
                    // this may already be promoted by PromoteTemps
                    PlaceContext::NonMutatingUse(Ctxt::Projection) => false,
                    PlaceContext::NonMutatingUse(_) => true,
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

    #[instrument(level = "debug", skip(self, rvalue))]
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        let Rvalue::Aggregate(kind, operands) = rvalue else {
            debug!("ignoring rvalue=`{rvalue:?}`");
            return;
        };

        let rvalue_ty = rvalue.ty(&self.ccx.body.local_decls, self.ccx.tcx);
        if !rvalue_ty.is_array() {
            debug!("ignoring rvalue=`{rvalue:?}`");
            if let Some(first) = operands.iter().next()
                && let Operand::Copy(place) | Operand::Move(place) = first
            {
                self.temps[place.local] = TempState::Unpromotable;
            }
            return;
        }

        debug!("pushing a candidate of type {:?} @ {:?}", kind, location);
        self.candidates.push(Candidate { location });
    }

    // #[instrument(level = "debug", skip(self, constant))]
    fn visit_constant(&mut self, constant: &ConstOperand<'tcx>, _location: Location) {
        if let Const::Unevaluated(c, _ty) = constant.const_
            && c.promoted.is_some()
        {
            self.already_promoted += 1;
        }

        // Skipping `super_constant` as the visitor is otherwise only looking for locals.
    }
}

fn collect_temps_and_candidates<'tcx>(
    ccx: &ConstCx<'_, 'tcx>,
) -> (IndexVec<Local, TempState>, Vec<Candidate>, usize) {
    let mut collector = Collector {
        temps: IndexVec::from_elem(TempState::Undefined, &ccx.body.local_decls),
        candidates: vec![],
        ccx,
        already_promoted: 0,
    };
    for (bb, data) in traversal::reverse_postorder(ccx.body) {
        collector.visit_basic_block_data(bb, data);
    }

    // debug!(collector.already_promoted);
    (collector.temps, collector.candidates, collector.already_promoted)
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
        let Some((place, rvalue @ Rvalue::Aggregate(box kind, operands))) =
            statement.kind.as_assign()
        else {
            bug!()
        };

        let TempState::Defined { .. } = self.temps[place.local] else {
            return Err(Unpromotable);
        };

        let AggregateKind::Array(arr_ty) = kind else {
            return Err(Unpromotable);
        };

        // lint only `if size_of(init) > STACK_THRESHOLD`
        let tcx = self.tcx;
        let rvalue_ty = rvalue.ty(self.body, tcx);
        if let Ok(layout) = tcx.layout_of(self.param_env.and(rvalue_ty))
            && let size = layout.layout.size()
            && size.bytes() <= STACK_THRESHOLD
        {
            debug!("size of array is too small: {:?}", size);
            return Err(Unpromotable);
        }

        if !arr_ty.is_trivially_pure_clone_copy() {
            return Err(Unpromotable);
        }

        for o in operands {
            self.validate_operand(o)?;
        }

        // No projections at all
        if !place.projection.is_empty() {
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

    // We can only promote interior borrows of promotable temps (non-temps
    // don't get promoted anyway).
    fn validate_local(&mut self, local: Local) -> Result<(), Unpromotable> {
        let TempState::Defined { location: loc, uses, valid } = self.temps[local] else {
            return Err(Unpromotable);
        };

        // We cannot promote things that need dropping, since the promoted value would not get
        // dropped.
        if self.qualif_local::<qualifs::NeedsDrop>(local) {
            return Err(Unpromotable);
        }

        if self.qualif_local::<qualifs::HasMutInterior>(local) {
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
            | ProjectionElem::Subslice { .. } => {}

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
                    && let Some(idx) = c.const_.try_eval_target_usize(self.tcx, self.param_env)
                    // Determine the type of the thing we are indexing.
                    && let ty::Array(_, len) = place_base.ty(self.body, self.tcx).ty.kind()
                    // It's an array; determine its length.
                    && let Some(len) = len.try_eval_target_usize(self.tcx, self.param_env)
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

    // The reference operation itself must be promotable.
    // (Needs to come after `validate_local` to avoid ICEs.)
    fn validate_ref(&mut self, kind: BorrowKind, place: &Place<'tcx>) -> Result<(), Unpromotable> {
        match kind {
            // Reject these borrow types just to be safe.
            // FIXME(RalfJung): could we allow them? Should we? No point in it until we have a usecase.
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
                    match len.try_eval_target_usize(self.tcx, self.param_env) {
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
            Rvalue::Use(operand) | Rvalue::Repeat(operand, _) => {
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
                    // Raw and fn pointer operations are not allowed inside consts and thus not promotable.
                    assert!(matches!(
                        op,
                        BinOp::Eq
                            | BinOp::Ne
                            | BinOp::Le
                            | BinOp::Lt
                            | BinOp::Ge
                            | BinOp::Gt
                            | BinOp::Offset
                    ));
                    return Err(Unpromotable);
                }

                match op {
                    BinOp::Div | BinOp::Rem => {
                        if lhs_ty.is_integral() {
                            let sz = lhs_ty.primitive_size(self.tcx);
                            // Integer division: the RHS must be a non-zero const.
                            let rhs_val = match rhs {
                                Operand::Constant(c) => {
                                    c.const_.try_eval_scalar_int(self.tcx, self.param_env)
                                }
                                _ => None,
                            };
                            match rhs_val.map(|x| x.assert_uint(sz)) {
                                // for the zero test, int vs uint does not matter
                                Some(x) if x != 0 => {}        // okay
                                _ => return Err(Unpromotable), // value not known or 0 -- not okay
                            }
                            // Furthermore, for signed divison, we also have to exclude `int::MIN / -1`.
                            if lhs_ty.is_signed() {
                                match rhs_val.map(|x| x.assert_int(sz)) {
                                    Some(-1) | None => {
                                        // The RHS is -1 or unknown, so we have to be careful.
                                        // But is the LHS int::MIN?
                                        let lhs_val = match lhs {
                                            Operand::Constant(c) => c
                                                .const_
                                                .try_eval_scalar_int(self.tcx, self.param_env),
                                            _ => None,
                                        };
                                        let lhs_min = sz.signed_int_min();
                                        match lhs_val.map(|x| x.assert_int(sz)) {
                                            Some(x) if x != lhs_min => {}  // okay
                                            _ => return Err(Unpromotable), // value not known or int::MIN -- not okay
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

            Rvalue::AddressOf(_, place) => {
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
            ty::FnDef(def_id, _) => self.tcx.is_const_fn_raw(def_id),
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
    candidates: &[Candidate],
) -> Vec<Candidate> {
    let mut validator = Validator { ccx, temps, promotion_safe_blocks: None };

    candidates
        .iter()
        .copied()
        .filter(|&candidate| validator.validate_candidate(candidate).is_ok())
        .collect()
}

struct Promoter<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source: &'a mut Body<'tcx>,
    promoted: Body<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,
    extra_statements: &'a mut Vec<(Location, Statement<'tcx>)>,

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
        self.promoted.basic_blocks_mut().push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo::outermost(span),
                kind: TerminatorKind::Return,
            }),
            is_cleanup: false,
        })
    }

    fn assign(&mut self, dest: Local, rvalue: Rvalue<'tcx>, span: Span) {
        let last = self.promoted.basic_blocks.last_index().unwrap();
        let data = &mut self.promoted[last];
        data.statements.push(Statement {
            source_info: SourceInfo::outermost(span),
            kind: StatementKind::Assign(Box::new((Place::from(dest), rvalue))),
        });
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
                let StatementKind::Assign(box (_place, rhs)) = &mut statement.kind else {
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
                    // This promoted involves a function call, so it may fail to evaluate.
                    // Let's make sure it is added to `required_consts` so that that failure cannot get lost.
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

    fn promote_candidate(mut self, candidate: Candidate, next_promoted_id: usize) -> Body<'tcx> {
        let def = self.source.source.def_id();
        let promoted_id = Promoted::new(next_promoted_id);
        let (mut rvalue, promoted_op, promoted_ref_rvalue) = {
            let promoted = &mut self.promoted;
            let tcx = self.tcx;
            let mut promoted_operand = |ty, span| {
                promoted.span = span;
                promoted.local_decls[RETURN_PLACE] = LocalDecl::new(ty, span);
                let args = tcx.erase_regions(GenericArgs::identity_for_item(tcx, def));
                let uneval = mir::UnevaluatedConst { def, args, promoted: Some(promoted_id) };

                ConstOperand { span, user_ty: None, const_: Const::Unevaluated(uneval, ty) }
            };

            let blocks = self.source.basic_blocks.as_mut();
            let local_decls = &mut self.source.local_decls;
            let loc = candidate.location;
            let statement = &mut blocks[loc.block].statements[loc.statement_index];
            let StatementKind::Assign(box (place, Rvalue::Aggregate(_kind, _operands))) =
                &mut statement.kind
            else {
                bug!()
            };

            let ty = local_decls[place.local].ty;
            let span = statement.source_info.span;

            let ref_ty = Ty::new_ref(tcx, tcx.lifetimes.re_erased, ty, hir::Mutability::Not);

            // Create a temp to hold the promoted reference.
            // This is because `*r` requires `r` to be a local,
            // otherwise we would use the `promoted` directly.
            let mut promoted_ref = LocalDecl::new(ref_ty, span);
            promoted_ref.source_info = statement.source_info;
            let promoted_ref: Local = local_decls.push(promoted_ref);
            // let new_state = TempState::Defined { location: loc, uses: 1, valid: Err(()) };
            let new_state = TempState::Unpromotable;
            assert_eq!(self.temps.push(new_state), promoted_ref);

            let promoted_operand = promoted_operand(ref_ty, span);
            let promoted_ref_statement = Statement {
                source_info: statement.source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(promoted_ref),
                    Rvalue::Use(Operand::Constant(Box::new(promoted_operand))),
                ))),
            };
            self.extra_statements.push((loc, promoted_ref_statement));

            let promoted_ref_place =
                Place { local: promoted_ref, projection: tcx.mk_place_elems(&[PlaceElem::Deref]) };

            // let promoted_rvalue = Rvalue::Use(Operand::Copy(place.clone()));
            let promoted_rvalue =
                Rvalue::Ref(tcx.lifetimes.re_erased, BorrowKind::Shared, Place::from(place.local));
            let promoted_ref_rvalue = Rvalue::Use(Operand::Copy(promoted_ref_place));

            (promoted_rvalue, promoted_operand, promoted_ref_rvalue)
        };

        let promoted_rvalue = promoted_ref_rvalue;

        assert_eq!(self.new_block(), START_BLOCK);
        self.visit_rvalue(
            &mut rvalue,
            Location { block: START_BLOCK, statement_index: usize::MAX },
        );

        {
            let blocks = self.source.basic_blocks_mut();
            let loc = candidate.location;
            let statement = &mut blocks[loc.block].statements[loc.statement_index];
            if let StatementKind::Assign(box (_place, ref mut rvalue)) = &mut statement.kind {
                *rvalue = promoted_rvalue;
            }
        }

        let span = self.promoted.span;
        self.assign(RETURN_PLACE, rvalue, span);

        // Now that we did promotion, we know whether we'll want to add this to `required_consts`.
        if self.add_to_required {
            self.source.required_consts.push(promoted_op);
        }

        self.promoted.source.promoted = Some(promoted_id);
        self.promoted
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_local(&mut self, local: &mut Local, _ctx: PlaceContext, _loc: Location) {
        if self.is_temp_kind(*local) {
            *local = self.promote_temp(*local);
        }
    }

    // #[instrument(level = "trace", skip(self, rvalue))]
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
    }

    fn visit_constant(&mut self, constant: &mut ConstOperand<'tcx>, _location: Location) {
        if constant.const_.is_required_const() {
            self.promoted.required_consts.push(*constant);
        }

        // Skipping `super_constant` as the visitor is otherwise only looking for locals.
    }
}

fn promote_candidates<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    mut temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
    already_promoted: usize,
) -> IndexVec<Promoted, Body<'tcx>> {
    // eagerly fail fast
    if candidates.is_empty() {
        return IndexVec::new();
    }

    let mut promotions = IndexVec::new();

    let mut extra_statements = vec![];
    // Visit candidates in reverse, in case they're nested.
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
        let initial_locals = IndexVec::from([LocalDecl::new(tcx.types.never, body.span)]);

        let mut scope = body.source_scopes[body.source_info(candidate.location).scope].clone();
        scope.parent_scope = None;

        let mut promoted = Body::new(
            /* source */ body.source, // `promoted` gets filled in below
            /* basic_blocks */ IndexVec::new(),
            /* source_scopes */ IndexVec::from([scope]),
            /* local_decls */ initial_locals,
            /* user_type_annotations */ IndexVec::new(),
            /* arg_count */ 0,
            /* var_debug_info */ vec![],
            /* span */ body.span,
            /* coroutine */ None,
            /* tainted_by_errors */ body.tainted_by_errors,
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
        };

        // `required_consts` of the promoted itself gets filled while building the MIR body.
        let promoted = promoter.promote_candidate(candidate, promotions.len() + already_promoted);
        // debug!(?promoted);
        promotions.push(promoted);
    }

    // Insert each of `extra_statements` before its indicated location, which
    // has to be done in reverse location order, to not invalidate the rest.
    extra_statements.sort_by_key(|&(loc, _)| cmp::Reverse(loc));
    for (loc, statement) in extra_statements {
        body[loc.block].statements.insert(loc.statement_index, statement);
    }

    // Eliminate assignments to, and drops of promoted temps.
    let is_promoted_out = |index: Local| temps[index] == TempState::PromotedOut;
    for block in body.basic_blocks_mut() {
        block.statements.retain(|statement| match &statement.kind {
            StatementKind::Assign(box (
                place,
                Rvalue::Use(Operand::Constant(box ConstOperand {
                    const_: Const::Val(_, ty), ..
                })),
            )) => {
                if ty.is_unit()
                    && let Some(index) = place.as_local()
                {
                    !is_promoted_out(index)
                } else {
                    true
                }
            }
            StatementKind::StorageLive(index) | StatementKind::StorageDead(index) => {
                !is_promoted_out(*index)
            }
            _ => true,
        });
    }

    promotions
}
