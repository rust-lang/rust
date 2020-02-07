//! The `Visitor` responsible for actually checking a `mir::Body` for invalid operations.

use rustc::middle::lang_items;
use rustc::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc::mir::*;
use rustc::traits::{self, TraitEngine};
use rustc::ty::cast::CastTy;
use rustc::ty::{self, adjustment::PointerCast, Predicate, Ty, TyCtxt};
use rustc_hir::{self as hir, def_id::DefId, HirId};
use rustc_index::bit_set::BitSet;
use rustc_span::symbol::sym;
use rustc_span::Span;

use std::ops::Deref;

use self::old_dataflow::IndirectlyMutableLocals;
use super::ops::{self, NonConstOp};
use super::qualifs::{self, HasMutInterior, NeedsDrop};
use super::resolver::FlowSensitiveAnalysis;
use super::{is_lang_panic_fn, ConstKind, Item, Qualif};
use crate::const_eval::{is_const_fn, is_min_const_fn, is_unstable_const_fn};
use crate::dataflow::{self as old_dataflow, generic as dataflow};

pub type IndirectlyMutableResults<'mir, 'tcx> =
    old_dataflow::DataflowResultsCursor<'mir, 'tcx, IndirectlyMutableLocals<'mir, 'tcx>>;

struct QualifCursor<'a, 'mir, 'tcx, Q: Qualif> {
    cursor: dataflow::ResultsCursor<'mir, 'tcx, FlowSensitiveAnalysis<'a, 'mir, 'tcx, Q>>,
    in_any_value_of_ty: BitSet<Local>,
}

impl<Q: Qualif> QualifCursor<'a, 'mir, 'tcx, Q> {
    pub fn new(q: Q, item: &'a Item<'mir, 'tcx>) -> Self {
        let analysis = FlowSensitiveAnalysis::new(q, item);
        let results = dataflow::Engine::new_generic(item.tcx, &item.body, item.def_id, analysis)
            .iterate_to_fixpoint();
        let cursor = dataflow::ResultsCursor::new(*item.body, results);

        let mut in_any_value_of_ty = BitSet::new_empty(item.body.local_decls.len());
        for (local, decl) in item.body.local_decls.iter_enumerated() {
            if Q::in_any_value_of_ty(item, decl.ty) {
                in_any_value_of_ty.insert(local);
            }
        }

        QualifCursor { cursor, in_any_value_of_ty }
    }
}

pub struct Qualifs<'a, 'mir, 'tcx> {
    has_mut_interior: QualifCursor<'a, 'mir, 'tcx, HasMutInterior>,
    needs_drop: QualifCursor<'a, 'mir, 'tcx, NeedsDrop>,
    indirectly_mutable: IndirectlyMutableResults<'mir, 'tcx>,
}

impl Qualifs<'a, 'mir, 'tcx> {
    fn indirectly_mutable(&mut self, local: Local, location: Location) -> bool {
        self.indirectly_mutable.seek(location);
        self.indirectly_mutable.get().contains(local)
    }

    /// Returns `true` if `local` is `NeedsDrop` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary
    fn needs_drop(&mut self, local: Local, location: Location) -> bool {
        if !self.needs_drop.in_any_value_of_ty.contains(local) {
            return false;
        }

        self.needs_drop.cursor.seek_before(location);
        self.needs_drop.cursor.get().contains(local) || self.indirectly_mutable(local, location)
    }

    /// Returns `true` if `local` is `HasMutInterior` at the given `Location`.
    ///
    /// Only updates the cursor if absolutely necessary.
    fn has_mut_interior(&mut self, local: Local, location: Location) -> bool {
        if !self.has_mut_interior.in_any_value_of_ty.contains(local) {
            return false;
        }

        self.has_mut_interior.cursor.seek_before(location);
        self.has_mut_interior.cursor.get().contains(local)
            || self.indirectly_mutable(local, location)
    }

    fn in_return_place(&mut self, item: &Item<'_, 'tcx>) -> ConstQualifs {
        // Find the `Return` terminator if one exists.
        //
        // If no `Return` terminator exists, this MIR is divergent. Just return the conservative
        // qualifs for the return type.
        let return_block = item
            .body
            .basic_blocks()
            .iter_enumerated()
            .find(|(_, block)| match block.terminator().kind {
                TerminatorKind::Return => true,
                _ => false,
            })
            .map(|(bb, _)| bb);

        let return_block = match return_block {
            None => return qualifs::in_any_value_of_ty(item, item.body.return_ty()),
            Some(bb) => bb,
        };

        let return_loc = item.body.terminator_loc(return_block);

        ConstQualifs {
            needs_drop: self.needs_drop(RETURN_PLACE, return_loc),
            has_mut_interior: self.has_mut_interior(RETURN_PLACE, return_loc),
        }
    }
}

pub struct Validator<'a, 'mir, 'tcx> {
    item: &'a Item<'mir, 'tcx>,
    qualifs: Qualifs<'a, 'mir, 'tcx>,

    /// The span of the current statement.
    span: Span,
}

impl Deref for Validator<'_, 'mir, 'tcx> {
    type Target = Item<'mir, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl Validator<'a, 'mir, 'tcx> {
    pub fn new(item: &'a Item<'mir, 'tcx>) -> Self {
        let needs_drop = QualifCursor::new(NeedsDrop, item);
        let has_mut_interior = QualifCursor::new(HasMutInterior, item);

        let dead_unwinds = BitSet::new_empty(item.body.basic_blocks().len());
        let indirectly_mutable = old_dataflow::do_dataflow(
            item.tcx,
            &*item.body,
            item.def_id,
            &item.tcx.get_attrs(item.def_id),
            &dead_unwinds,
            old_dataflow::IndirectlyMutableLocals::new(item.tcx, *item.body, item.param_env),
            |_, local| old_dataflow::DebugFormatted::new(&local),
        );

        let indirectly_mutable =
            old_dataflow::DataflowResultsCursor::new(indirectly_mutable, *item.body);

        let qualifs = Qualifs { needs_drop, has_mut_interior, indirectly_mutable };

        Validator { span: item.body.span, item, qualifs }
    }

    pub fn check_item(&mut self) {
        let Item { tcx, body, def_id, const_kind, .. } = *self.item;

        // The local type and predicate checks are remnants from the old `min_const_fn`
        // pass, so they only run on `const fn`s.
        if const_kind == Some(ConstKind::ConstFn) {
            self.check_item_predicates();

            for local in &body.local_decls {
                self.span = local.source_info.span;
                self.check_local_or_return_ty(local.ty);
            }

            // impl trait is gone in MIR, so check the return type of a const fn by its signature
            // instead of the type of the return place.
            self.span = body.local_decls[RETURN_PLACE].source_info.span;
            let return_ty = tcx.fn_sig(def_id).output();
            self.check_local_or_return_ty(return_ty.skip_binder());
        }

        check_short_circuiting_in_const_local(self.item);

        if body.is_cfg_cyclic() {
            // We can't provide a good span for the error here, but this should be caught by the
            // HIR const-checker anyways.
            self.check_op_spanned(ops::Loop, body.span);
        }

        self.visit_body(body);

        // Ensure that the end result is `Sync` in a non-thread local `static`.
        let should_check_for_sync =
            const_kind == Some(ConstKind::Static) && !tcx.has_attr(def_id, sym::thread_local);

        if should_check_for_sync {
            let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
            check_return_ty_is_sync(tcx, &body, hir_id);
        }
    }

    pub fn qualifs_in_return_place(&mut self) -> ConstQualifs {
        self.qualifs.in_return_place(self.item)
    }

    /// Emits an error at the given `span` if an expression cannot be evaluated in the current
    /// context.
    pub fn check_op_spanned<O>(&mut self, op: O, span: Span)
    where
        O: NonConstOp,
    {
        debug!("check_op: op={:?}", op);

        if op.is_allowed_in_item(self) {
            return;
        }

        // If an operation is supported in miri (and is not already controlled by a feature gate) it
        // can be turned on with `-Zunleash-the-miri-inside-of-you`.
        let is_unleashable = O::IS_SUPPORTED_IN_MIRI && O::feature_gate().is_none();
        if is_unleashable && self.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
            self.tcx.sess.span_warn(span, "skipping const checks");
            return;
        }

        op.emit_error(self, span);
    }

    /// Emits an error if an expression cannot be evaluated in the current context.
    pub fn check_op(&mut self, op: impl NonConstOp) {
        let span = self.span;
        self.check_op_spanned(op, span)
    }

    fn check_static(&mut self, def_id: DefId, span: Span) {
        let is_thread_local = self.tcx.has_attr(def_id, sym::thread_local);
        if is_thread_local {
            self.check_op_spanned(ops::ThreadLocalAccess, span)
        } else {
            self.check_op_spanned(ops::StaticAccess, span)
        }
    }

    fn check_local_or_return_ty(&mut self, ty: Ty<'tcx>) {
        for ty in ty.walk() {
            match ty.kind {
                ty::Ref(_, _, hir::Mutability::Mut) => self.check_op(ops::MutBorrow),
                ty::Opaque(..) => self.check_op(ops::ImplTrait),
                ty::FnPtr(..) => self.check_op(ops::FnPtr),

                ty::Dynamic(preds, _) => {
                    for pred in preds.iter() {
                        match pred.skip_binder() {
                            ty::ExistentialPredicate::AutoTrait(_)
                            | ty::ExistentialPredicate::Projection(_) => {
                                self.check_op(ops::TraitBound(hir::Constness::Const))
                            }
                            ty::ExistentialPredicate::Trait(trait_ref) => {
                                if Some(trait_ref.def_id) != self.tcx.lang_items().sized_trait() {
                                    self.check_op(ops::TraitBound(hir::Constness::Const))
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn check_item_predicates(&mut self) {
        let Item { tcx, def_id, .. } = *self.item;

        let mut current = def_id;
        loop {
            let predicates = tcx.predicates_of(current);
            for (predicate, _) in predicates.predicates {
                match predicate {
                    Predicate::RegionOutlives(_)
                    | Predicate::TypeOutlives(_)
                    | Predicate::WellFormed(_)
                    | Predicate::Projection(_)
                    | Predicate::ConstEvaluatable(..) => continue,
                    Predicate::ObjectSafe(_) => {
                        bug!("object safe predicate on function: {:#?}", predicate)
                    }
                    Predicate::ClosureKind(..) => {
                        bug!("closure kind predicate on function: {:#?}", predicate)
                    }
                    Predicate::Subtype(_) => {
                        bug!("subtype predicate on function: {:#?}", predicate)
                    }
                    Predicate::Trait(pred, constness) => {
                        if Some(pred.def_id()) == tcx.lang_items().sized_trait() {
                            continue;
                        }
                        match pred.skip_binder().self_ty().kind {
                            ty::Param(ref p) => {
                                let generics = tcx.generics_of(current);
                                let def = generics.type_param(p, tcx);
                                let span = tcx.def_span(def.def_id);

                                self.check_op_spanned(ops::TraitBound(*constness), span);
                            }
                            // other kinds of bounds are either tautologies
                            // or cause errors in other passes
                            _ => continue,
                        }
                    }
                }
            }
            match predicates.parent {
                Some(parent) => current = parent,
                None => break,
            }
        }
    }
}

impl Visitor<'tcx> for Validator<'_, 'mir, 'tcx> {
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
            Rvalue::Ref(_, kind, ref place) => {
                if let Some(reborrowed_proj) = place_as_reborrow(self.tcx, *self.body, place) {
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
                    self.visit_place_base(&place.local, ctx, location);
                    self.visit_projection(&place.local, reborrowed_proj, ctx, location);
                    return;
                }
            }
            Rvalue::AddressOf(mutbl, ref place) => {
                if let Some(reborrowed_proj) = place_as_reborrow(self.tcx, *self.body, place) {
                    let ctx = match mutbl {
                        Mutability::Not => {
                            PlaceContext::NonMutatingUse(NonMutatingUseContext::AddressOf)
                        }
                        Mutability::Mut => PlaceContext::MutatingUse(MutatingUseContext::AddressOf),
                    };
                    self.visit_place_base(&place.local, ctx, location);
                    self.visit_projection(&place.local, reborrowed_proj, ctx, location);
                    return;
                }
            }
            _ => {}
        }

        self.super_rvalue(rvalue, location);

        match *rvalue {
            Rvalue::Use(_)
            | Rvalue::Repeat(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Len(_)
            | Rvalue::Aggregate(..) => {}

            // `&mut` and `&raw mut`
            Rvalue::AddressOf(Mutability::Mut, _) => self.check_op(ops::MutBorrow),

            Rvalue::Ref(_, BorrowKind::Mut { .. }, ref place)
            | Rvalue::Ref(_, BorrowKind::Unique, ref place) => {
                let ty = place.ty(*self.body, self.tcx).ty;
                let is_allowed = match ty.kind {
                    // Inside a `static mut`, `&mut [...]` is allowed.
                    ty::Array(..) | ty::Slice(_) if self.const_kind() == ConstKind::StaticMut => {
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
                    self.check_op(ops::MutBorrow);
                }
            }

            Rvalue::Ref(_, BorrowKind::Shared, ref place)
            | Rvalue::Ref(_, BorrowKind::Shallow, ref place)
            | Rvalue::AddressOf(Mutability::Not, ref place) => {
                let borrowed_place_has_mut_interior = HasMutInterior::in_place(
                    &self.item,
                    &mut |local| self.qualifs.has_mut_interior(local, location),
                    place.as_ref(),
                );

                if borrowed_place_has_mut_interior {
                    self.check_op(ops::CellBorrow);
                }
            }

            Rvalue::Cast(CastKind::Pointer(PointerCast::UnsafeFnPointer), ..)
            | Rvalue::Cast(CastKind::Pointer(PointerCast::ClosureFnPointer(_)), ..)
            | Rvalue::Cast(CastKind::Pointer(PointerCast::ReifyFnPointer), ..) => {
                self.check_op(ops::FnPtr)
            }

            Rvalue::Cast(CastKind::Pointer(PointerCast::MutToConstPointer), ..)
            | Rvalue::Cast(CastKind::Pointer(PointerCast::ArrayToPointer), ..) => {}

            Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), ..) => {
                self.check_op(ops::UnsizingCast)
            }

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(*self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");

                if let (CastTy::Ptr(_), CastTy::Int(_)) | (CastTy::FnPtr, CastTy::Int(_)) =
                    (cast_in, cast_out)
                {
                    self.check_op(ops::RawPtrToIntCast);
                }
            }

            Rvalue::NullaryOp(NullOp::SizeOf, _) => {}
            Rvalue::NullaryOp(NullOp::Box, _) => self.check_op(ops::HeapAllocation),

            Rvalue::UnaryOp(op, ref place) => {
                let ty = place.ty(*self.body, self.tcx);
                match ty.kind {
                    // Operations on the following primitive types are always allowed.
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) => {}

                    // All other operations are forbidden
                    _ => self.check_op(ops::Arithmetic(op.into(), ty)),
                }
            }

            Rvalue::CheckedBinaryOp(op, ref lhs, _) | Rvalue::BinaryOp(op, ref lhs, _) => {
                let ty = lhs.ty(*self.body, self.tcx);
                match ty.kind {
                    ty::RawPtr(_) | ty::FnPtr(..) => {
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

                    // Operations on the following primitive types are always allowed.
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) => {}

                    // All other operations are forbidden
                    _ => self.check_op(ops::Arithmetic(op.into(), ty)),
                }
            }
        }
    }

    fn visit_place_base(&mut self, place_local: &Local, context: PlaceContext, location: Location) {
        trace!(
            "visit_place_base: place_local={:?} context={:?} location={:?}",
            place_local,
            context,
            location,
        );
        self.super_place_base(place_local, context, location);
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
        place_local: &Local,
        proj_base: &[PlaceElem<'tcx>],
        elem: &PlaceElem<'tcx>,
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
                let base_ty = Place::ty_from(*place_local, proj_base, *self.body, self.tcx).ty;
                if let ty::RawPtr(_) = base_ty.kind {
                    if proj_base.is_empty() {
                        if let (local, []) = (place_local, proj_base) {
                            let decl = &self.body.local_decls[*local];
                            if let LocalInfo::StaticRef { def_id, .. } = decl.local_info {
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
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Field(..)
            | ProjectionElem::Index(_) => {
                let base_ty = Place::ty_from(*place_local, proj_base, *self.body, self.tcx).ty;
                match base_ty.ty_adt_def() {
                    Some(def) if def.is_union() => {
                        self.check_op(ops::UnionAccess);
                    }

                    _ => {}
                }
            }

            ProjectionElem::Downcast(..) => {
                self.check_op(ops::Downcast);
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

            StatementKind::InlineAsm { .. } => self.check_op(ops::InlineAsm),

            // Try to ensure that no `match` expressions have gotten through the HIR const-checker.
            StatementKind::FakeRead(FakeReadCause::ForMatchGuard, _)
            | StatementKind::FakeRead(FakeReadCause::ForGuardBinding, _)
            | StatementKind::FakeRead(FakeReadCause::ForMatchedPlace, _) => {
                self.check_op(ops::IfOrMatch)
            }

            // Other `FakeRead`s are allowed
            StatementKind::FakeRead(FakeReadCause::ForLet, _)
            | StatementKind::FakeRead(FakeReadCause::ForIndex, _) => {}

            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag { .. }
            | StatementKind::AscribeUserType(..)
            | StatementKind::Nop => {}
        }
    }

    fn visit_terminator_kind(&mut self, kind: &TerminatorKind<'tcx>, location: Location) {
        trace!("visit_terminator_kind: kind={:?} location={:?}", kind, location);
        self.super_terminator_kind(kind, location);

        match kind {
            TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Goto { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Resume => {}

            // Emitted when, e.g., doing integer division to detect division by zero.
            TerminatorKind::Assert { .. } => {}

            TerminatorKind::SwitchInt { .. } => self.check_op(ops::IfOrMatch),
            TerminatorKind::Abort => self.check_op(ops::Abort),
            TerminatorKind::GeneratorDrop | TerminatorKind::Yield { .. } => {
                self.check_op(ops::Generator)
            }

            TerminatorKind::Call { func, .. } => {
                let fn_ty = func.ty(*self.body, self.tcx);
                let callee = match fn_ty.kind {
                    ty::FnDef(def_id, _) => def_id,

                    ty::FnPtr(_) => {
                        self.check_op(ops::FnCallIndirect);
                        return;
                    }
                    _ => {
                        self.check_op(ops::FnCallOther);
                        return;
                    }
                };

                let Item { tcx, def_id: caller, .. } = *self.item;

                if self.const_kind == Some(ConstKind::ConstFn)
                    && is_min_const_fn(tcx, caller)
                    && !is_min_const_fn(tcx, callee)
                    && is_unstable_const_fn(tcx, callee)
                        .map_or(true, |feature| !self.span.allows_unstable(feature))
                {
                    tcx.sess
                        .span_err(self.span, "Stable `const fn`s cannot call unstable `const fn`s");
                }

                if is_const_fn(tcx, callee) {
                    return;
                }

                if is_lang_panic_fn(tcx, callee) {
                    self.check_op(ops::Panic);
                } else if let Some(feature) = is_unstable_const_fn(tcx, callee) {
                    // Exempt unstable const fns inside of macros with
                    // `#[allow_internal_unstable]`.
                    if !self.span.allows_unstable(feature) {
                        self.check_op(ops::FnCallUnstable(callee, feature));
                    }
                } else {
                    self.check_op(ops::FnCallNonConst(callee));
                }
            }

            // Forbid all `Drop` terminators unless the place being dropped is a local with no
            // projections that cannot be `NeedsDrop`.
            TerminatorKind::Drop { location: dropped_place, .. }
            | TerminatorKind::DropAndReplace { location: dropped_place, .. } => {
                let mut err_span = self.span;

                // Check to see if the type of this place can ever have a drop impl. If not, this
                // `Drop` terminator is frivolous.
                let ty_needs_drop =
                    dropped_place.ty(*self.body, self.tcx).ty.needs_drop(self.tcx, self.param_env);

                if !ty_needs_drop {
                    return;
                }

                let needs_drop = if let Some(local) = dropped_place.as_local() {
                    // Use the span where the local was declared as the span of the drop error.
                    err_span = self.body.local_decls[local].source_info.span;
                    self.qualifs.needs_drop(local, location)
                } else {
                    true
                };

                if needs_drop {
                    self.check_op_spanned(ops::LiveDrop, err_span);
                }
            }
        }
    }
}

fn check_short_circuiting_in_const_local(item: &Item<'_, 'tcx>) {
    let body = item.body;

    if body.control_flow_destroyed.is_empty() {
        return;
    }

    let mut locals = body.vars_iter();
    if let Some(local) = locals.next() {
        let span = body.local_decls[local].source_info.span;
        let mut error = item.tcx.sess.struct_span_err(
            span,
            &format!(
                "new features like let bindings are not permitted in {}s \
                which also use short circuiting operators",
                item.const_kind(),
            ),
        );
        for (span, kind) in body.control_flow_destroyed.iter() {
            error.span_note(
                *span,
                &format!(
                    "use of {} here does not actually short circuit due to \
                the const evaluator presently not being able to do control flow. \
                See https://github.com/rust-lang/rust/issues/49146 for more \
                information.",
                    kind
                ),
            );
        }
        for local in locals {
            let span = body.local_decls[local].source_info.span;
            error.span_note(span, "more locals are defined here");
        }
        error.emit();
    }
}

fn check_return_ty_is_sync(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, hir_id: HirId) {
    let ty = body.return_ty();
    tcx.infer_ctxt().enter(|infcx| {
        let cause = traits::ObligationCause::new(body.span, hir_id, traits::SharedStatic);
        let mut fulfillment_cx = traits::FulfillmentContext::new();
        let sync_def_id = tcx.require_lang_item(lang_items::SyncTraitLangItem, Some(body.span));
        fulfillment_cx.register_bound(&infcx, ty::ParamEnv::empty(), ty, sync_def_id, cause);
        if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(&err, None, false);
        }
    });
}

fn place_as_reborrow(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    place: &'a Place<'tcx>,
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
        match inner_ty.kind {
            ty::Ref(..) => Some(inner),
            _ => None,
        }
    })
}
