//! The `Visitor` responsible for actually checking a `mir::Body` for invalid operations.

use rustc_errors::struct_span_err;
use rustc_hir::{self as hir, LangItem};
use rustc_hir::{def_id::DefId, HirId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::{self, Instance, InstanceDef, TyCtxt};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt;
use rustc_trait_selection::traits::{self, TraitEngine};

use std::borrow::Cow;
use std::ops::Deref;

use super::ops::{self, NonConstOp};
use super::qualifs::{self, CustomEq, HasMutInterior, NeedsDrop};
use super::resolver::FlowSensitiveAnalysis;
use super::{is_lang_panic_fn, ConstCx, Qualif};
use crate::const_eval::{is_const_fn, is_unstable_const_fn};
use crate::dataflow::impls::MaybeMutBorrowedLocals;
use crate::dataflow::{self, Analysis};

// We are using `MaybeMutBorrowedLocals` as a proxy for whether an item may have been mutated
// through a pointer prior to the given point. This is okay even though `MaybeMutBorrowedLocals`
// kills locals upon `StorageDead` because a local will never be used after a `StorageDead`.
type IndirectlyMutableResults<'mir, 'tcx> =
    dataflow::ResultsCursor<'mir, 'tcx, MaybeMutBorrowedLocals<'mir, 'tcx>>;

type QualifResults<'mir, 'tcx, Q> =
    dataflow::ResultsCursor<'mir, 'tcx, FlowSensitiveAnalysis<'mir, 'mir, 'tcx, Q>>;

#[derive(Default)]
pub struct Qualifs<'mir, 'tcx> {
    has_mut_interior: Option<QualifResults<'mir, 'tcx, HasMutInterior>>,
    needs_drop: Option<QualifResults<'mir, 'tcx, NeedsDrop>>,
    indirectly_mutable: Option<IndirectlyMutableResults<'mir, 'tcx>>,
}

impl Qualifs<'mir, 'tcx> {
    pub fn indirectly_mutable(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        local: Local,
        location: Location,
    ) -> bool {
        let indirectly_mutable = self.indirectly_mutable.get_or_insert_with(|| {
            let ConstCx { tcx, body, def_id, param_env, .. } = *ccx;

            // We can use `unsound_ignore_borrow_on_drop` here because custom drop impls are not
            // allowed in a const.
            //
            // FIXME(ecstaticmorse): Someday we want to allow custom drop impls. How do we do this
            // without breaking stable code?
            MaybeMutBorrowedLocals::mut_borrows_only(tcx, &body, param_env)
                .unsound_ignore_borrow_on_drop()
                .into_engine(tcx, &body, def_id.to_def_id())
                .iterate_to_fixpoint()
                .into_results_cursor(&body)
        });

        indirectly_mutable.seek_before_primary_effect(location);
        indirectly_mutable.get().contains(local)
    }

    /// Returns `true` if `local` is `NeedsDrop` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary
    pub fn needs_drop(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        local: Local,
        location: Location,
    ) -> bool {
        let ty = ccx.body.local_decls[local].ty;
        if !NeedsDrop::in_any_value_of_ty(ccx, ty) {
            return false;
        }

        let needs_drop = self.needs_drop.get_or_insert_with(|| {
            let ConstCx { tcx, body, def_id, .. } = *ccx;

            FlowSensitiveAnalysis::new(NeedsDrop, ccx)
                .into_engine(tcx, &body, def_id.to_def_id())
                .iterate_to_fixpoint()
                .into_results_cursor(&body)
        });

        needs_drop.seek_before_primary_effect(location);
        needs_drop.get().contains(local) || self.indirectly_mutable(ccx, local, location)
    }

    /// Returns `true` if `local` is `HasMutInterior` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary.
    pub fn has_mut_interior(
        &mut self,
        ccx: &'mir ConstCx<'mir, 'tcx>,
        local: Local,
        location: Location,
    ) -> bool {
        let ty = ccx.body.local_decls[local].ty;
        if !HasMutInterior::in_any_value_of_ty(ccx, ty) {
            return false;
        }

        let has_mut_interior = self.has_mut_interior.get_or_insert_with(|| {
            let ConstCx { tcx, body, def_id, .. } = *ccx;

            FlowSensitiveAnalysis::new(HasMutInterior, ccx)
                .into_engine(tcx, &body, def_id.to_def_id())
                .iterate_to_fixpoint()
                .into_results_cursor(&body)
        });

        has_mut_interior.seek_before_primary_effect(location);
        has_mut_interior.get().contains(local) || self.indirectly_mutable(ccx, local, location)
    }

    fn in_return_place(&mut self, ccx: &'mir ConstCx<'mir, 'tcx>) -> ConstQualifs {
        // Find the `Return` terminator if one exists.
        //
        // If no `Return` terminator exists, this MIR is divergent. Just return the conservative
        // qualifs for the return type.
        let return_block = ccx
            .body
            .basic_blocks()
            .iter_enumerated()
            .find(|(_, block)| match block.terminator().kind {
                TerminatorKind::Return => true,
                _ => false,
            })
            .map(|(bb, _)| bb);

        let return_block = match return_block {
            None => return qualifs::in_any_value_of_ty(ccx, ccx.body.return_ty()),
            Some(bb) => bb,
        };

        let return_loc = ccx.body.terminator_loc(return_block);

        let custom_eq = match ccx.const_kind() {
            // We don't care whether a `const fn` returns a value that is not structurally
            // matchable. Functions calls are opaque and always use type-based qualification, so
            // this value should never be used.
            hir::ConstContext::ConstFn => true,

            // If we know that all values of the return type are structurally matchable, there's no
            // need to run dataflow.
            _ if !CustomEq::in_any_value_of_ty(ccx, ccx.body.return_ty()) => false,

            hir::ConstContext::Const | hir::ConstContext::Static(_) => {
                let mut cursor = FlowSensitiveAnalysis::new(CustomEq, ccx)
                    .into_engine(ccx.tcx, &ccx.body, ccx.def_id.to_def_id())
                    .iterate_to_fixpoint()
                    .into_results_cursor(&ccx.body);

                cursor.seek_after_primary_effect(return_loc);
                cursor.contains(RETURN_PLACE)
            }
        };

        ConstQualifs {
            needs_drop: self.needs_drop(ccx, RETURN_PLACE, return_loc),
            has_mut_interior: self.has_mut_interior(ccx, RETURN_PLACE, return_loc),
            custom_eq,
        }
    }
}

pub struct Validator<'mir, 'tcx> {
    ccx: &'mir ConstCx<'mir, 'tcx>,
    qualifs: Qualifs<'mir, 'tcx>,

    /// The span of the current statement.
    span: Span,
}

impl Deref for Validator<'mir, 'tcx> {
    type Target = ConstCx<'mir, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.ccx
    }
}

impl Validator<'mir, 'tcx> {
    pub fn new(ccx: &'mir ConstCx<'mir, 'tcx>) -> Self {
        Validator { span: ccx.body.span, ccx, qualifs: Default::default() }
    }

    pub fn check_body(&mut self) {
        let ConstCx { tcx, body, def_id, const_kind, .. } = *self.ccx;

        let use_min_const_fn_checks = (const_kind == Some(hir::ConstContext::ConstFn)
            && crate::const_eval::is_min_const_fn(tcx, def_id.to_def_id()))
            && !tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you;

        if use_min_const_fn_checks {
            // Enforce `min_const_fn` for stable `const fn`s.
            use crate::transform::qualify_min_const_fn::is_min_const_fn;
            if let Err((span, err)) = is_min_const_fn(tcx, def_id.to_def_id(), &body) {
                error_min_const_fn_violation(tcx, span, err);
                return;
            }
        }

        self.visit_body(&body);

        // Ensure that the end result is `Sync` in a non-thread local `static`.
        let should_check_for_sync = const_kind
            == Some(hir::ConstContext::Static(hir::Mutability::Not))
            && !tcx.is_thread_local_static(def_id.to_def_id());

        if should_check_for_sync {
            let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
            check_return_ty_is_sync(tcx, &body, hir_id);
        }
    }

    pub fn qualifs_in_return_place(&mut self) -> ConstQualifs {
        self.qualifs.in_return_place(self.ccx)
    }

    /// Emits an error if an expression cannot be evaluated in the current context.
    pub fn check_op(&mut self, op: impl NonConstOp) {
        ops::non_const(self.ccx, op, self.span);
    }

    /// Emits an error at the given `span` if an expression cannot be evaluated in the current
    /// context.
    pub fn check_op_spanned(&mut self, op: impl NonConstOp, span: Span) {
        ops::non_const(self.ccx, op, span);
    }

    fn check_static(&mut self, def_id: DefId, span: Span) {
        assert!(
            !self.tcx.is_thread_local_static(def_id),
            "tls access is checked in `Rvalue::ThreadLocalRef"
        );
        self.check_op_spanned(ops::StaticAccess, span)
    }
}

impl Visitor<'tcx> for Validator<'mir, 'tcx> {
    fn visit_basic_block_data(&mut self, bb: BasicBlock, block: &BasicBlockData<'tcx>) {
        trace!("visit_basic_block_data: bb={:?} is_cleanup={:?}", bb, block.is_cleanup);

        // Just as the old checker did, we skip const-checking basic blocks on the unwind path.
        // These blocks often drop locals that would otherwise be returned from the function.
        //
        // FIXME: This shouldn't be unsound since a panic at compile time will cause a compiler
        // error anyway, but maybe we should do more here?
        if block.is_cleanup {
            return;
        }

        self.super_basic_block_data(bb, block);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        trace!("visit_rvalue: rvalue={:?} location={:?}", rvalue, location);

        // Special-case reborrows to be more like a copy of a reference.
        match *rvalue {
            Rvalue::Ref(_, kind, place) => {
                if let Some(reborrowed_proj) = place_as_reborrow(self.tcx, self.body, place) {
                    let ctx = match kind {
                        BorrowKind::Shared => {
                            PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow)
                        }
                        BorrowKind::Shallow => {
                            PlaceContext::NonMutatingUse(NonMutatingUseContext::ShallowBorrow)
                        }
                        BorrowKind::Unique => {
                            PlaceContext::NonMutatingUse(NonMutatingUseContext::UniqueBorrow)
                        }
                        BorrowKind::Mut { .. } => {
                            PlaceContext::MutatingUse(MutatingUseContext::Borrow)
                        }
                    };
                    self.visit_local(&place.local, ctx, location);
                    self.visit_projection(place.local, reborrowed_proj, ctx, location);
                    return;
                }
            }
            Rvalue::AddressOf(mutbl, place) => {
                if let Some(reborrowed_proj) = place_as_reborrow(self.tcx, self.body, place) {
                    let ctx = match mutbl {
                        Mutability::Not => {
                            PlaceContext::NonMutatingUse(NonMutatingUseContext::AddressOf)
                        }
                        Mutability::Mut => PlaceContext::MutatingUse(MutatingUseContext::AddressOf),
                    };
                    self.visit_local(&place.local, ctx, location);
                    self.visit_projection(place.local, reborrowed_proj, ctx, location);
                    return;
                }
            }
            _ => {}
        }

        self.super_rvalue(rvalue, location);

        match *rvalue {
            Rvalue::ThreadLocalRef(_) => self.check_op(ops::ThreadLocalAccess),

            Rvalue::Use(_)
            | Rvalue::Repeat(..)
            | Rvalue::UnaryOp(UnOp::Neg, _)
            | Rvalue::UnaryOp(UnOp::Not, _)
            | Rvalue::NullaryOp(NullOp::SizeOf, _)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::Cast(CastKind::Pointer(_), ..)
            | Rvalue::Discriminant(..)
            | Rvalue::Len(_)
            | Rvalue::Aggregate(..) => {}

            Rvalue::Ref(_, kind @ BorrowKind::Mut { .. }, ref place)
            | Rvalue::Ref(_, kind @ BorrowKind::Unique, ref place) => {
                let ty = place.ty(self.body, self.tcx).ty;
                let is_allowed = match ty.kind() {
                    // Inside a `static mut`, `&mut [...]` is allowed.
                    ty::Array(..) | ty::Slice(_)
                        if self.const_kind() == hir::ConstContext::Static(hir::Mutability::Mut) =>
                    {
                        true
                    }

                    // FIXME(ecstaticmorse): We could allow `&mut []` inside a const context given
                    // that this is merely a ZST and it is already eligible for promotion.
                    // This may require an RFC?
                    /*
                    ty::Array(_, len) if len.try_eval_usize(cx.tcx, cx.param_env) == Some(0)
                        => true,
                    */
                    _ => false,
                };

                if !is_allowed {
                    if let BorrowKind::Mut { .. } = kind {
                        self.check_op(ops::MutBorrow);
                    } else {
                        self.check_op(ops::CellBorrow);
                    }
                }
            }

            Rvalue::AddressOf(Mutability::Mut, _) => self.check_op(ops::MutAddressOf),

            Rvalue::Ref(_, BorrowKind::Shared | BorrowKind::Shallow, ref place)
            | Rvalue::AddressOf(Mutability::Not, ref place) => {
                let borrowed_place_has_mut_interior = qualifs::in_place::<HasMutInterior, _>(
                    &self.ccx,
                    &mut |local| self.qualifs.has_mut_interior(self.ccx, local, location),
                    place.as_ref(),
                );

                if borrowed_place_has_mut_interior {
                    self.check_op(ops::CellBorrow);
                }
            }

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");

                if let (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Int(_)) = (cast_in, cast_out) {
                    self.check_op(ops::RawPtrToIntCast);
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.body, self.tcx).kind() {
                    assert!(
                        op == BinOp::Eq
                            || op == BinOp::Ne
                            || op == BinOp::Le
                            || op == BinOp::Lt
                            || op == BinOp::Ge
                            || op == BinOp::Gt
                            || op == BinOp::Offset
                    );

                    self.check_op(ops::RawPtrComparison);
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => {
                self.check_op(ops::HeapAllocation);
            }
        }
    }

    fn visit_operand(&mut self, op: &Operand<'tcx>, location: Location) {
        self.super_operand(op, location);
        if let Operand::Constant(c) = op {
            if let Some(def_id) = c.check_static_ptr(self.tcx) {
                self.check_static(def_id, self.span);
            }
        }
    }
    fn visit_projection_elem(
        &mut self,
        place_local: Local,
        proj_base: &[PlaceElem<'tcx>],
        elem: PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        trace!(
            "visit_projection_elem: place_local={:?} proj_base={:?} elem={:?} \
            context={:?} location={:?}",
            place_local,
            proj_base,
            elem,
            context,
            location,
        );

        self.super_projection_elem(place_local, proj_base, elem, context, location);

        match elem {
            ProjectionElem::Deref => {
                let base_ty = Place::ty_from(place_local, proj_base, self.body, self.tcx).ty;
                if let ty::RawPtr(_) = base_ty.kind() {
                    if proj_base.is_empty() {
                        if let (local, []) = (place_local, proj_base) {
                            let decl = &self.body.local_decls[local];
                            if let Some(box LocalInfo::StaticRef { def_id, .. }) = decl.local_info {
                                let span = decl.source_info.span;
                                self.check_static(def_id, span);
                                return;
                            }
                        }
                    }
                    self.check_op(ops::RawPtrDeref);
                }

                if context.is_mutating_use() {
                    self.check_op(ops::MutDeref);
                }
            }

            ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Downcast(..)
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Field(..)
            | ProjectionElem::Index(_) => {
                let base_ty = Place::ty_from(place_local, proj_base, self.body, self.tcx).ty;
                match base_ty.ty_adt_def() {
                    Some(def) if def.is_union() => {
                        self.check_op(ops::UnionAccess);
                    }

                    _ => {}
                }
            }
        }
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        trace!("visit_source_info: source_info={:?}", source_info);
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        trace!("visit_statement: statement={:?} location={:?}", statement, location);

        match statement.kind {
            StatementKind::Assign(..) | StatementKind::SetDiscriminant { .. } => {
                self.super_statement(statement, location);
            }

            StatementKind::LlvmInlineAsm { .. } => {
                self.super_statement(statement, location);
                self.check_op(ops::InlineAsm);
            }

            StatementKind::FakeRead(..)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag { .. }
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::Nop => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        trace!("visit_terminator: terminator={:?} location={:?}", terminator, location);
        self.super_terminator(terminator, location);

        match &terminator.kind {
            TerminatorKind::Call { func, .. } => {
                let fn_ty = func.ty(self.body, self.tcx);

                let (def_id, substs) = match *fn_ty.kind() {
                    ty::FnDef(def_id, substs) => (def_id, substs),

                    ty::FnPtr(_) => {
                        self.check_op(ops::FnCallIndirect);
                        return;
                    }
                    _ => {
                        span_bug!(terminator.source_info.span, "invalid callee of type {:?}", fn_ty)
                    }
                };

                // At this point, we are calling a function whose `DefId` is known...
                if is_const_fn(self.tcx, def_id) {
                    return;
                }

                // See if this is a trait method for a concrete type whose impl of that trait is
                // `const`.
                if self.tcx.features().const_trait_impl {
                    let instance = Instance::resolve(self.tcx, self.param_env, def_id, substs);
                    debug!("Resolving ({:?}) -> {:?}", def_id, instance);
                    if let Ok(Some(func)) = instance {
                        if let InstanceDef::Item(def) = func.def {
                            if is_const_fn(self.tcx, def.did) {
                                return;
                            }
                        }
                    }
                }

                if is_lang_panic_fn(self.tcx, def_id) {
                    self.check_op(ops::Panic);
                } else if let Some(feature) = is_unstable_const_fn(self.tcx, def_id) {
                    // Exempt unstable const fns inside of macros or functions with
                    // `#[allow_internal_unstable]`.
                    use crate::transform::qualify_min_const_fn::lib_feature_allowed;
                    if !self.span.allows_unstable(feature)
                        && !lib_feature_allowed(self.tcx, self.def_id.to_def_id(), feature)
                    {
                        self.check_op(ops::FnCallUnstable(def_id, feature));
                    }
                } else {
                    self.check_op(ops::FnCallNonConst(def_id));
                }
            }

            // Forbid all `Drop` terminators unless the place being dropped is a local with no
            // projections that cannot be `NeedsDrop`.
            TerminatorKind::Drop { place: dropped_place, .. }
            | TerminatorKind::DropAndReplace { place: dropped_place, .. } => {
                // If we are checking live drops after drop-elaboration, don't emit duplicate
                // errors here.
                if super::post_drop_elaboration::checking_enabled(self.tcx) {
                    return;
                }

                let mut err_span = self.span;

                // Check to see if the type of this place can ever have a drop impl. If not, this
                // `Drop` terminator is frivolous.
                let ty_needs_drop =
                    dropped_place.ty(self.body, self.tcx).ty.needs_drop(self.tcx, self.param_env);

                if !ty_needs_drop {
                    return;
                }

                let needs_drop = if let Some(local) = dropped_place.as_local() {
                    // Use the span where the local was declared as the span of the drop error.
                    err_span = self.body.local_decls[local].source_info.span;
                    self.qualifs.needs_drop(self.ccx, local, location)
                } else {
                    true
                };

                if needs_drop {
                    self.check_op_spanned(
                        ops::LiveDrop(Some(terminator.source_info.span)),
                        err_span,
                    );
                }
            }

            TerminatorKind::InlineAsm { .. } => {
                self.check_op(ops::InlineAsm);
            }

            // FIXME: Some of these are only caught by `min_const_fn`, but should error here
            // instead.
            TerminatorKind::Abort
            | TerminatorKind::Assert { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => {}
        }
    }
}

fn error_min_const_fn_violation(tcx: TyCtxt<'_>, span: Span, msg: Cow<'_, str>) {
    struct_span_err!(tcx.sess, span, E0723, "{}", msg)
        .note(
            "see issue #57563 <https://github.com/rust-lang/rust/issues/57563> \
             for more information",
        )
        .help("add `#![feature(const_fn)]` to the crate attributes to enable")
        .emit();
}

fn check_return_ty_is_sync(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, hir_id: HirId) {
    let ty = body.return_ty();
    tcx.infer_ctxt().enter(|infcx| {
        let cause = traits::ObligationCause::new(body.span, hir_id, traits::SharedStatic);
        let mut fulfillment_cx = traits::FulfillmentContext::new();
        let sync_def_id = tcx.require_lang_item(LangItem::Sync, Some(body.span));
        fulfillment_cx.register_bound(&infcx, ty::ParamEnv::empty(), ty, sync_def_id, cause);
        if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(&err, None, false);
        }
    });
}

fn place_as_reborrow(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Option<&'a [PlaceElem<'tcx>]> {
    place.projection.split_last().and_then(|(outermost, inner)| {
        if outermost != &ProjectionElem::Deref {
            return None;
        }

        // A borrow of a `static` also looks like `&(*_1)` in the MIR, but `_1` is a `const`
        // that points to the allocation for the static. Don't treat these as reborrows.
        if body.local_decls[place.local].is_ref_to_static() {
            return None;
        }

        // Ensure the type being derefed is a reference and not a raw pointer.
        //
        // This is sufficient to prevent an access to a `static mut` from being marked as a
        // reborrow, even if the check above were to disappear.
        let inner_ty = Place::ty_from(place.local, inner, body, tcx).ty;
        match inner_ty.kind() {
            ty::Ref(..) => Some(inner),
            _ => None,
        }
    })
}
