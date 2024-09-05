//! Inference of closure parameter types based on the closure's expected type.

use std::{cmp, convert::Infallible, mem};

use chalk_ir::{
    cast::Cast,
    fold::{FallibleTypeFolder, TypeFoldable},
    BoundVar, DebruijnIndex, FnSubst, Mutability, TyKind,
};
use either::Either;
use hir_def::{
    data::adt::VariantData,
    hir::{
        Array, AsmOperand, BinaryOp, BindingId, CaptureBy, Expr, ExprId, Pat, PatId, Statement,
        UnaryOp,
    },
    lang_item::LangItem,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId, FieldId, HasModule, TupleFieldId, TupleId, VariantId,
};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};
use stdx::{format_to, never};
use syntax::utils::is_raw_identifier;

use crate::{
    db::{HirDatabase, InternedClosure},
    error_lifetime, from_chalk_trait_id, from_placeholder_idx,
    generics::Generics,
    make_binders,
    mir::{BorrowKind, MirSpan, MutBorrowKind, ProjectionElem},
    to_chalk_trait_id,
    traits::FnTrait,
    utils::{self, elaborate_clause_supertraits},
    Adjust, Adjustment, AliasEq, AliasTy, Binders, BindingMode, ChalkTraitId, ClosureId, DynTy,
    DynTyExt, FnAbi, FnPointer, FnSig, Interner, OpaqueTy, ProjectionTyExt, Substitution, Ty,
    TyExt, WhereClause,
};

use super::{Expectation, InferenceContext};

impl InferenceContext<'_> {
    // This function handles both closures and coroutines.
    pub(super) fn deduce_closure_type_from_expectations(
        &mut self,
        closure_expr: ExprId,
        closure_ty: &Ty,
        sig_ty: &Ty,
        expectation: &Expectation,
    ) {
        let expected_ty = match expectation.to_option(&mut self.table) {
            Some(ty) => ty,
            None => return,
        };

        if let TyKind::Closure(closure_id, _) = closure_ty.kind(Interner) {
            if let Some(closure_kind) = self.deduce_closure_kind_from_expectations(&expected_ty) {
                self.result
                    .closure_info
                    .entry(*closure_id)
                    .or_insert_with(|| (Vec::new(), closure_kind));
            }
        }

        // Deduction from where-clauses in scope, as well as fn-pointer coercion are handled here.
        let _ = self.coerce(Some(closure_expr), closure_ty, &expected_ty);

        // Coroutines are not Fn* so return early.
        if matches!(closure_ty.kind(Interner), TyKind::Coroutine(..)) {
            return;
        }

        // Deduction based on the expected `dyn Fn` is done separately.
        if let TyKind::Dyn(dyn_ty) = expected_ty.kind(Interner) {
            if let Some(sig) = self.deduce_sig_from_dyn_ty(dyn_ty) {
                let expected_sig_ty = TyKind::Function(sig).intern(Interner);

                self.unify(sig_ty, &expected_sig_ty);
            }
        }
    }

    // Closure kind deductions are mostly from `rustc_hir_typeck/src/closure.rs`.
    // Might need to port closure sig deductions too.
    fn deduce_closure_kind_from_expectations(&mut self, expected_ty: &Ty) -> Option<FnTrait> {
        match expected_ty.kind(Interner) {
            TyKind::Alias(AliasTy::Opaque(OpaqueTy { .. })) | TyKind::OpaqueType(..) => {
                let clauses = expected_ty
                    .impl_trait_bounds(self.db)
                    .into_iter()
                    .flatten()
                    .map(|b| b.into_value_and_skipped_binders().0);
                self.deduce_closure_kind_from_predicate_clauses(clauses)
            }
            TyKind::Dyn(dyn_ty) => dyn_ty.principal().and_then(|trait_ref| {
                self.fn_trait_kind_from_trait_id(from_chalk_trait_id(trait_ref.trait_id))
            }),
            TyKind::InferenceVar(ty, chalk_ir::TyVariableKind::General) => {
                let clauses = self.clauses_for_self_ty(*ty);
                self.deduce_closure_kind_from_predicate_clauses(clauses.into_iter())
            }
            TyKind::Function(_) => Some(FnTrait::Fn),
            _ => None,
        }
    }

    fn deduce_closure_kind_from_predicate_clauses(
        &self,
        clauses: impl DoubleEndedIterator<Item = WhereClause>,
    ) -> Option<FnTrait> {
        let mut expected_kind = None;

        for clause in elaborate_clause_supertraits(self.db, clauses.rev()) {
            let trait_id = match clause {
                WhereClause::AliasEq(AliasEq {
                    alias: AliasTy::Projection(projection), ..
                }) => Some(projection.trait_(self.db)),
                WhereClause::Implemented(trait_ref) => {
                    Some(from_chalk_trait_id(trait_ref.trait_id))
                }
                _ => None,
            };
            if let Some(closure_kind) =
                trait_id.and_then(|trait_id| self.fn_trait_kind_from_trait_id(trait_id))
            {
                // `FnX`'s variants order is opposite from rustc, so use `cmp::max` instead of `cmp::min`
                expected_kind = Some(
                    expected_kind
                        .map_or_else(|| closure_kind, |current| cmp::max(current, closure_kind)),
                );
            }
        }

        expected_kind
    }

    fn deduce_sig_from_dyn_ty(&self, dyn_ty: &DynTy) -> Option<FnPointer> {
        // Search for a predicate like `<$self as FnX<Args>>::Output == Ret`

        let fn_traits: SmallVec<[ChalkTraitId; 3]> =
            utils::fn_traits(self.db.upcast(), self.owner.module(self.db.upcast()).krate())
                .map(to_chalk_trait_id)
                .collect();

        let self_ty = self.result.standard_types.unknown.clone();
        let bounds = dyn_ty.bounds.clone().substitute(Interner, &[self_ty.cast(Interner)]);
        for bound in bounds.iter(Interner) {
            // NOTE(skip_binders): the extracted types are rebound by the returned `FnPointer`
            if let WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection), ty }) =
                bound.skip_binders()
            {
                let assoc_data = self.db.associated_ty_data(projection.associated_ty_id);
                if !fn_traits.contains(&assoc_data.trait_id) {
                    return None;
                }

                // Skip `Self`, get the type argument.
                let arg = projection.substitution.as_slice(Interner).get(1)?;
                if let Some(subst) = arg.ty(Interner)?.as_tuple() {
                    let generic_args = subst.as_slice(Interner);
                    let mut sig_tys = Vec::with_capacity(generic_args.len() + 1);
                    for arg in generic_args {
                        sig_tys.push(arg.ty(Interner)?.clone());
                    }
                    sig_tys.push(ty.clone());

                    cov_mark::hit!(dyn_fn_param_informs_call_site_closure_signature);
                    return Some(FnPointer {
                        num_binders: bound.len(Interner),
                        sig: FnSig {
                            abi: FnAbi::RustCall,
                            safety: chalk_ir::Safety::Safe,
                            variadic: false,
                        },
                        substitution: FnSubst(Substitution::from_iter(Interner, sig_tys)),
                    });
                }
            }
        }

        None
    }

    fn fn_trait_kind_from_trait_id(&self, trait_id: hir_def::TraitId) -> Option<FnTrait> {
        FnTrait::from_lang_item(self.db.lang_attr(trait_id.into())?)
    }
}

// The below functions handle capture and closure kind (Fn, FnMut, ..)

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct HirPlace {
    pub(crate) local: BindingId,
    pub(crate) projections: Vec<ProjectionElem<Infallible, Ty>>,
}

impl HirPlace {
    fn ty(&self, ctx: &mut InferenceContext<'_>) -> Ty {
        let mut ty = ctx.table.resolve_completely(ctx.result[self.local].clone());
        for p in &self.projections {
            ty = p.projected_ty(
                ty,
                ctx.db,
                |_, _, _| {
                    unreachable!("Closure field only happens in MIR");
                },
                ctx.owner.module(ctx.db.upcast()).krate(),
            );
        }
        ty
    }

    fn capture_kind_of_truncated_place(
        &self,
        mut current_capture: CaptureKind,
        len: usize,
    ) -> CaptureKind {
        if let CaptureKind::ByRef(BorrowKind::Mut {
            kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow,
        }) = current_capture
        {
            if self.projections[len..].iter().any(|it| *it == ProjectionElem::Deref) {
                current_capture =
                    CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture });
            }
        }
        current_capture
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureKind {
    ByRef(BorrowKind),
    ByValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedItem {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    ///
    /// Even though we always report only the last span (i.e. the most inclusive span),
    /// we need to keep them all, since when a closure occurs inside a closure, we
    /// copy all captures of the inner closure to the outer closure, and then we may
    /// truncate them, and we want the correct span to be reported.
    span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
    pub(crate) ty: Binders<Ty>,
}

impl CapturedItem {
    pub fn local(&self) -> BindingId {
        self.place.local
    }

    /// Returns whether this place has any field (aka. non-deref) projections.
    pub fn has_field_projections(&self) -> bool {
        self.place.projections.iter().any(|it| !matches!(it, ProjectionElem::Deref))
    }

    pub fn ty(&self, subst: &Substitution) -> Ty {
        self.ty.clone().substitute(Interner, utils::ClosureSubst(subst).parent_subst())
    }

    pub fn kind(&self) -> CaptureKind {
        self.kind
    }

    pub fn spans(&self) -> SmallVec<[MirSpan; 3]> {
        self.span_stacks.iter().map(|stack| *stack.last().expect("empty span stack")).collect()
    }

    /// Converts the place to a name that can be inserted into source code.
    pub fn place_to_name(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let mut result = body[self.place.local].name.unescaped().display(db.upcast()).to_string();
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {}
                ProjectionElem::Field(Either::Left(f)) => {
                    match &*f.parent.variant_data(db.upcast()) {
                        VariantData::Record(fields) => {
                            result.push('_');
                            result.push_str(fields[f.local_id].name.as_str())
                        }
                        VariantData::Tuple(fields) => {
                            let index = fields.iter().position(|it| it.0 == f.local_id);
                            if let Some(index) = index {
                                format_to!(result, "_{index}");
                            }
                        }
                        VariantData::Unit => {}
                    }
                }
                ProjectionElem::Field(Either::Right(f)) => format_to!(result, "_{}", f.index),
                &ProjectionElem::ClosureField(field) => format_to!(result, "_{field}"),
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        if is_raw_identifier(&result, db.crate_graph()[owner.module(db.upcast()).krate()].edition) {
            result.insert_str(0, "r#");
        }
        result
    }

    pub fn display_place_source_code(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db.upcast());
        let edition = db.crate_graph()[krate].edition;
        let mut result = body[self.place.local].name.display(db.upcast(), edition).to_string();
        for proj in &self.place.projections {
            match proj {
                // In source code autoderef kicks in.
                ProjectionElem::Deref => {}
                ProjectionElem::Field(Either::Left(f)) => {
                    let variant_data = f.parent.variant_data(db.upcast());
                    match &*variant_data {
                        VariantData::Record(fields) => format_to!(
                            result,
                            ".{}",
                            fields[f.local_id].name.display(db.upcast(), edition)
                        ),
                        VariantData::Tuple(fields) => format_to!(
                            result,
                            ".{}",
                            fields.iter().position(|it| it.0 == f.local_id).unwrap_or_default()
                        ),
                        VariantData::Unit => {}
                    }
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let field = f.index;
                    format_to!(result, ".{field}");
                }
                &ProjectionElem::ClosureField(field) => {
                    format_to!(result, ".{field}");
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        let final_derefs_count = self
            .place
            .projections
            .iter()
            .rev()
            .take_while(|proj| matches!(proj, ProjectionElem::Deref))
            .count();
        result.insert_str(0, &"*".repeat(final_derefs_count));
        result
    }

    pub fn display_place(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db.upcast());
        let edition = db.crate_graph()[krate].edition;
        let mut result = body[self.place.local].name.display(db.upcast(), edition).to_string();
        let mut field_need_paren = false;
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {
                    result = format!("*{result}");
                    field_need_paren = true;
                }
                ProjectionElem::Field(Either::Left(f)) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    let variant_data = f.parent.variant_data(db.upcast());
                    let field = match &*variant_data {
                        VariantData::Record(fields) => fields[f.local_id].name.as_str().to_owned(),
                        VariantData::Tuple(fields) => fields
                            .iter()
                            .position(|it| it.0 == f.local_id)
                            .unwrap_or_default()
                            .to_string(),
                        VariantData::Unit => "[missing field]".to_owned(),
                    };
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let field = f.index;
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                &ProjectionElem::ClosureField(field) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapturedItemWithoutTy {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    pub(crate) span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
}

impl CapturedItemWithoutTy {
    fn with_ty(self, ctx: &mut InferenceContext<'_>) -> CapturedItem {
        let ty = self.place.ty(ctx);
        let ty = match &self.kind {
            CaptureKind::ByValue => ty,
            CaptureKind::ByRef(bk) => {
                let m = match bk {
                    BorrowKind::Mut { .. } => Mutability::Mut,
                    _ => Mutability::Not,
                };
                TyKind::Ref(m, error_lifetime(), ty).intern(Interner)
            }
        };
        return CapturedItem {
            place: self.place,
            kind: self.kind,
            span_stacks: self.span_stacks,
            ty: replace_placeholder_with_binder(ctx, ty),
        };

        fn replace_placeholder_with_binder(ctx: &mut InferenceContext<'_>, ty: Ty) -> Binders<Ty> {
            struct Filler<'a> {
                db: &'a dyn HirDatabase,
                generics: &'a Generics,
            }
            impl FallibleTypeFolder<Interner> for Filler<'_> {
                type Error = ();

                fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = Self::Error> {
                    self
                }

                fn interner(&self) -> Interner {
                    Interner
                }

                fn try_fold_free_placeholder_const(
                    &mut self,
                    ty: chalk_ir::Ty<Interner>,
                    idx: chalk_ir::PlaceholderIndex,
                    outer_binder: DebruijnIndex,
                ) -> Result<chalk_ir::Const<Interner>, Self::Error> {
                    let x = from_placeholder_idx(self.db, idx);
                    let Some(idx) = self.generics.type_or_const_param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_const(Interner, ty))
                }

                fn try_fold_free_placeholder_ty(
                    &mut self,
                    idx: chalk_ir::PlaceholderIndex,
                    outer_binder: DebruijnIndex,
                ) -> std::result::Result<Ty, Self::Error> {
                    let x = from_placeholder_idx(self.db, idx);
                    let Some(idx) = self.generics.type_or_const_param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_ty(Interner))
                }
            }
            let Some(generics) = ctx.generics() else {
                return Binders::empty(Interner, ty);
            };
            let filler = &mut Filler { db: ctx.db, generics };
            let result = ty.clone().try_fold_with(filler, DebruijnIndex::INNERMOST).unwrap_or(ty);
            make_binders(ctx.db, filler.generics, result)
        }
    }
}

impl InferenceContext<'_> {
    fn place_of_expr(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        let r = self.place_of_expr_without_adjust(tgt_expr)?;
        let default = vec![];
        let adjustments = self.result.expr_adjustments.get(&tgt_expr).unwrap_or(&default);
        apply_adjusts_to_place(&mut self.current_capture_span_stack, r, adjustments)
    }

    /// Changes `current_capture_span_stack` to contain the stack of spans for this expr.
    fn place_of_expr_without_adjust(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        self.current_capture_span_stack.clear();
        match &self.body[tgt_expr] {
            Expr::Path(p) => {
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, tgt_expr);
                if let Some(ResolveValueResult::ValueNs(ValueNs::LocalBinding(b), _)) =
                    resolver.resolve_path_in_value_ns(self.db.upcast(), p)
                {
                    self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                    return Some(HirPlace { local: b, projections: vec![] });
                }
            }
            Expr::Field { expr, name: _ } => {
                let mut place = self.place_of_expr(*expr)?;
                let field = self.result.field_resolution(tgt_expr)?;
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                place.projections.push(ProjectionElem::Field(field));
                return Some(place);
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if matches!(
                    self.expr_ty_after_adjustments(*expr).kind(Interner),
                    TyKind::Ref(..) | TyKind::Raw(..)
                ) {
                    let mut place = self.place_of_expr(*expr)?;
                    self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                    place.projections.push(ProjectionElem::Deref);
                    return Some(place);
                }
            }
            _ => (),
        }
        None
    }

    fn push_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        self.current_captures.push(CapturedItemWithoutTy {
            place,
            kind,
            span_stacks: smallvec![self.current_capture_span_stack.iter().copied().collect()],
        });
    }

    fn truncate_capture_spans(&self, capture: &mut CapturedItemWithoutTy, mut truncate_to: usize) {
        // The first span is the identifier, and it must always remain.
        truncate_to += 1;
        for span_stack in &mut capture.span_stacks {
            let mut remained = truncate_to;
            let mut actual_truncate_to = 0;
            for &span in &*span_stack {
                actual_truncate_to += 1;
                if !span.is_ref_span(self.body) {
                    remained -= 1;
                    if remained == 0 {
                        break;
                    }
                }
            }
            if actual_truncate_to < span_stack.len()
                && span_stack[actual_truncate_to].is_ref_span(self.body)
            {
                // Include the ref operator if there is one, we will fix it later (in `strip_captures_ref_span()`) if it's incorrect.
                actual_truncate_to += 1;
            }
            span_stack.truncate(actual_truncate_to);
        }
    }

    fn ref_expr(&mut self, expr: ExprId, place: Option<HirPlace>) {
        if let Some(place) = place {
            self.add_capture(place, CaptureKind::ByRef(BorrowKind::Shared));
        }
        self.walk_expr(expr);
    }

    fn add_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        if self.is_upvar(&place) {
            self.push_capture(place, kind);
        }
    }

    fn mutate_expr(&mut self, expr: ExprId, place: Option<HirPlace>) {
        if let Some(place) = place {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            );
        }
        self.walk_expr(expr);
    }

    fn consume_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.consume_place(place);
        }
        self.walk_expr(expr);
    }

    fn consume_place(&mut self, place: HirPlace) {
        if self.is_upvar(&place) {
            let ty = place.ty(self);
            let kind = if self.is_ty_copy(ty) {
                CaptureKind::ByRef(BorrowKind::Shared)
            } else {
                CaptureKind::ByValue
            };
            self.push_capture(place, kind);
        }
    }

    fn walk_expr_with_adjust(&mut self, tgt_expr: ExprId, adjustment: &[Adjustment]) {
        if let Some((last, rest)) = adjustment.split_last() {
            match &last.kind {
                Adjust::NeverToAny | Adjust::Deref(None) | Adjust::Pointer(_) => {
                    self.walk_expr_with_adjust(tgt_expr, rest)
                }
                Adjust::Deref(Some(m)) => match m.0 {
                    Some(m) => {
                        self.ref_capture_with_adjusts(m, tgt_expr, rest);
                    }
                    None => unreachable!(),
                },
                Adjust::Borrow(b) => {
                    self.ref_capture_with_adjusts(b.mutability(), tgt_expr, rest);
                }
            }
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn ref_capture_with_adjusts(&mut self, m: Mutability, tgt_expr: ExprId, rest: &[Adjustment]) {
        let capture_kind = match m {
            Mutability::Mut => CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            Mutability::Not => CaptureKind::ByRef(BorrowKind::Shared),
        };
        if let Some(place) = self.place_of_expr_without_adjust(tgt_expr) {
            if let Some(place) =
                apply_adjusts_to_place(&mut self.current_capture_span_stack, place, rest)
            {
                self.add_capture(place, capture_kind);
            }
        }
        self.walk_expr_with_adjust(tgt_expr, rest);
    }

    fn walk_expr(&mut self, tgt_expr: ExprId) {
        if let Some(it) = self.result.expr_adjustments.get_mut(&tgt_expr) {
            // FIXME: this take is completely unneeded, and just is here to make borrow checker
            // happy. Remove it if you can.
            let x_taken = mem::take(it);
            self.walk_expr_with_adjust(tgt_expr, &x_taken);
            *self.result.expr_adjustments.get_mut(&tgt_expr).unwrap() = x_taken;
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn walk_expr_without_adjust(&mut self, tgt_expr: ExprId) {
        match &self.body[tgt_expr] {
            Expr::OffsetOf(_) => (),
            Expr::InlineAsm(e) => e.operands.iter().for_each(|(_, op)| match op {
                AsmOperand::In { expr, .. }
                | AsmOperand::Out { expr: Some(expr), .. }
                | AsmOperand::InOut { expr, .. } => self.walk_expr_without_adjust(*expr),
                AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    self.walk_expr_without_adjust(*in_expr);
                    if let Some(out_expr) = out_expr {
                        self.walk_expr_without_adjust(*out_expr);
                    }
                }
                AsmOperand::Out { expr: None, .. }
                | AsmOperand::Const(_)
                | AsmOperand::Label(_)
                | AsmOperand::Sym(_) => (),
            }),
            Expr::If { condition, then_branch, else_branch } => {
                self.consume_expr(*condition);
                self.consume_expr(*then_branch);
                if let &Some(expr) = else_branch {
                    self.consume_expr(expr);
                }
            }
            Expr::Async { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Block { statements, tail, .. } => {
                for s in statements.iter() {
                    match s {
                        Statement::Let { pat, type_ref: _, initializer, else_branch } => {
                            if let Some(else_branch) = else_branch {
                                self.consume_expr(*else_branch);
                            }
                            if let Some(initializer) = initializer {
                                if else_branch.is_some() {
                                    self.consume_expr(*initializer);
                                } else {
                                    self.walk_expr(*initializer);
                                }
                                if let Some(place) = self.place_of_expr(*initializer) {
                                    self.consume_with_pat(place, *pat);
                                }
                            }
                        }
                        Statement::Expr { expr, has_semi: _ } => {
                            self.consume_expr(*expr);
                        }
                        Statement::Item => (),
                    }
                }
                if let Some(tail) = tail {
                    self.consume_expr(*tail);
                }
            }
            Expr::Call { callee, args, is_assignee_expr: _ } => {
                self.consume_expr(*callee);
                self.consume_exprs(args.iter().copied());
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.consume_expr(*receiver);
                self.consume_exprs(args.iter().copied());
            }
            Expr::Match { expr, arms } => {
                for arm in arms.iter() {
                    self.consume_expr(arm.expr);
                    if let Some(guard) = arm.guard {
                        self.consume_expr(guard);
                    }
                }
                self.walk_expr(*expr);
                if let Some(discr_place) = self.place_of_expr(*expr) {
                    if self.is_upvar(&discr_place) {
                        let mut capture_mode = None;
                        for arm in arms.iter() {
                            self.walk_pat(&mut capture_mode, arm.pat);
                        }
                        if let Some(c) = capture_mode {
                            self.push_capture(discr_place, c);
                        }
                    }
                }
            }
            Expr::Break { expr, label: _ }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    self.consume_expr(expr);
                }
            }
            &Expr::Become { expr } => {
                self.consume_expr(expr);
            }
            Expr::RecordLit { fields, spread, .. } => {
                if let &Some(expr) = spread {
                    self.consume_expr(expr);
                }
                self.consume_exprs(fields.iter().map(|it| it.expr));
            }
            Expr::Field { expr, name: _ } => self.select_from_expr(*expr),
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if matches!(
                    self.expr_ty_after_adjustments(*expr).kind(Interner),
                    TyKind::Ref(..) | TyKind::Raw(..)
                ) {
                    self.select_from_expr(*expr);
                } else if let Some((f, _)) = self.result.method_resolution(tgt_expr) {
                    let mutability = 'b: {
                        if let Some(deref_trait) =
                            self.resolve_lang_item(LangItem::DerefMut).and_then(|it| it.as_trait())
                        {
                            if let Some(deref_fn) = self
                                .db
                                .trait_data(deref_trait)
                                .method_by_name(&Name::new_symbol_root(sym::deref_mut.clone()))
                            {
                                break 'b deref_fn == f;
                            }
                        }
                        false
                    };
                    let place = self.place_of_expr(*expr);
                    if mutability {
                        self.mutate_expr(*expr, place);
                    } else {
                        self.ref_expr(*expr, place);
                    }
                } else {
                    self.select_from_expr(*expr);
                }
            }
            Expr::UnaryOp { expr, op: _ }
            | Expr::Array(Array::Repeat { initializer: expr, repeat: _ })
            | Expr::Await { expr }
            | Expr::Loop { body: expr, label: _ }
            | Expr::Let { pat: _, expr }
            | Expr::Box { expr }
            | Expr::Cast { expr, type_ref: _ } => {
                self.consume_expr(*expr);
            }
            Expr::Ref { expr, rawness: _, mutability } => {
                // We need to do this before we push the span so the order will be correct.
                let place = self.place_of_expr(*expr);
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                match mutability {
                    hir_def::type_ref::Mutability::Shared => self.ref_expr(*expr, place),
                    hir_def::type_ref::Mutability::Mut => self.mutate_expr(*expr, place),
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => {
                let Some(op) = op else {
                    return;
                };
                if matches!(op, BinaryOp::Assignment { .. }) {
                    let place = self.place_of_expr(*lhs);
                    self.mutate_expr(*lhs, place);
                    self.consume_expr(*rhs);
                    return;
                }
                self.consume_expr(*lhs);
                self.consume_expr(*rhs);
            }
            Expr::Range { lhs, rhs, range_type: _ } => {
                if let &Some(expr) = lhs {
                    self.consume_expr(expr);
                }
                if let &Some(expr) = rhs {
                    self.consume_expr(expr);
                }
            }
            Expr::Index { base, index, is_assignee_expr: _ } => {
                self.select_from_expr(*base);
                self.consume_expr(*index);
            }
            Expr::Closure { .. } => {
                let ty = self.expr_ty(tgt_expr);
                let TyKind::Closure(id, _) = ty.kind(Interner) else {
                    never!("closure type is always closure");
                    return;
                };
                let (captures, _) =
                    self.result.closure_info.get(id).expect(
                        "We sort closures, so we should always have data for inner closures",
                    );
                let mut cc = mem::take(&mut self.current_captures);
                cc.extend(captures.iter().filter(|it| self.is_upvar(&it.place)).map(|it| {
                    CapturedItemWithoutTy {
                        place: it.place.clone(),
                        kind: it.kind,
                        span_stacks: it.span_stacks.clone(),
                    }
                }));
                self.current_captures = cc;
            }
            Expr::Array(Array::ElementList { elements: exprs, is_assignee_expr: _ })
            | Expr::Tuple { exprs, is_assignee_expr: _ } => {
                self.consume_exprs(exprs.iter().copied())
            }

            Expr::Missing
            | Expr::Continue { .. }
            | Expr::Path(_)
            | Expr::Literal(_)
            | Expr::Const(_)
            | Expr::Underscore => (),
        }
    }

    fn walk_pat(&mut self, result: &mut Option<CaptureKind>, pat: PatId) {
        let mut update_result = |ck: CaptureKind| match result {
            Some(r) => {
                *r = cmp::max(*r, ck);
            }
            None => *result = Some(ck),
        };

        self.walk_pat_inner(
            pat,
            &mut update_result,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
        );
    }

    fn walk_pat_inner(
        &mut self,
        p: PatId,
        update_result: &mut impl FnMut(CaptureKind),
        mut for_mut: BorrowKind,
    ) {
        match &self.body[p] {
            Pat::Ref { .. }
            | Pat::Box { .. }
            | Pat::Missing
            | Pat::Wild
            | Pat::Tuple { .. }
            | Pat::Or(_) => (),
            Pat::TupleStruct { .. } | Pat::Record { .. } => {
                if let Some(variant) = self.result.variant_resolution_for_pat(p) {
                    let adt = variant.adt_id(self.db.upcast());
                    let is_multivariant = match adt {
                        hir_def::AdtId::EnumId(e) => self.db.enum_data(e).variants.len() != 1,
                        _ => false,
                    };
                    if is_multivariant {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    }
                }
            }
            Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Path(_)
            | Pat::Lit(_)
            | Pat::Range { .. } => {
                update_result(CaptureKind::ByRef(BorrowKind::Shared));
            }
            Pat::Bind { id, .. } => match self.result.binding_modes[p] {
                crate::BindingMode::Move => {
                    if self.is_ty_copy(self.result.type_of_binding[*id].clone()) {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    } else {
                        update_result(CaptureKind::ByValue);
                    }
                }
                crate::BindingMode::Ref(r) => match r {
                    Mutability::Mut => update_result(CaptureKind::ByRef(for_mut)),
                    Mutability::Not => update_result(CaptureKind::ByRef(BorrowKind::Shared)),
                },
            },
        }
        if self.result.pat_adjustments.get(&p).map_or(false, |it| !it.is_empty()) {
            for_mut = BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture };
        }
        self.body.walk_pats_shallow(p, |p| self.walk_pat_inner(p, update_result, for_mut));
    }

    fn expr_ty(&self, expr: ExprId) -> Ty {
        self.result[expr].clone()
    }

    fn expr_ty_after_adjustments(&self, e: ExprId) -> Ty {
        let mut ty = None;
        if let Some(it) = self.result.expr_adjustments.get(&e) {
            if let Some(it) = it.last() {
                ty = Some(it.target.clone());
            }
        }
        ty.unwrap_or_else(|| self.expr_ty(e))
    }

    fn is_upvar(&self, place: &HirPlace) -> bool {
        if let Some(c) = self.current_closure {
            let InternedClosure(_, root) = self.db.lookup_intern_closure(c.into());
            return self.body.is_binding_upvar(place.local, root);
        }
        false
    }

    fn is_ty_copy(&mut self, ty: Ty) -> bool {
        if let TyKind::Closure(id, _) = ty.kind(Interner) {
            // FIXME: We handle closure as a special case, since chalk consider every closure as copy. We
            // should probably let chalk know which closures are copy, but I don't know how doing it
            // without creating query cycles.
            return self.result.closure_info.get(id).map(|it| it.1 == FnTrait::Fn).unwrap_or(true);
        }
        self.table.resolve_completely(ty).is_copy(self.db, self.owner)
    }

    fn select_from_expr(&mut self, expr: ExprId) {
        self.walk_expr(expr);
    }

    fn restrict_precision_for_unsafe(&mut self) {
        // FIXME: Borrow checker problems without this.
        let mut current_captures = std::mem::take(&mut self.current_captures);
        for capture in &mut current_captures {
            let mut ty = self.table.resolve_completely(self.result[capture.place.local].clone());
            if ty.as_raw_ptr().is_some() || ty.is_union() {
                capture.kind = CaptureKind::ByRef(BorrowKind::Shared);
                self.truncate_capture_spans(capture, 0);
                capture.place.projections.truncate(0);
                continue;
            }
            for (i, p) in capture.place.projections.iter().enumerate() {
                ty = p.projected_ty(
                    ty,
                    self.db,
                    |_, _, _| {
                        unreachable!("Closure field only happens in MIR");
                    },
                    self.owner.module(self.db.upcast()).krate(),
                );
                if ty.as_raw_ptr().is_some() || ty.is_union() {
                    capture.kind = CaptureKind::ByRef(BorrowKind::Shared);
                    self.truncate_capture_spans(capture, i + 1);
                    capture.place.projections.truncate(i + 1);
                    break;
                }
            }
        }
        self.current_captures = current_captures;
    }

    fn adjust_for_move_closure(&mut self) {
        // FIXME: Borrow checker won't allow without this.
        let mut current_captures = std::mem::take(&mut self.current_captures);
        for capture in &mut current_captures {
            if let Some(first_deref) =
                capture.place.projections.iter().position(|proj| *proj == ProjectionElem::Deref)
            {
                self.truncate_capture_spans(capture, first_deref);
                capture.place.projections.truncate(first_deref);
            }
            capture.kind = CaptureKind::ByValue;
        }
        self.current_captures = current_captures;
    }

    fn minimize_captures(&mut self) {
        self.current_captures.sort_unstable_by_key(|it| it.place.projections.len());
        let mut hash_map = FxHashMap::<HirPlace, usize>::default();
        let result = mem::take(&mut self.current_captures);
        for mut item in result {
            let mut lookup_place = HirPlace { local: item.place.local, projections: vec![] };
            let mut it = item.place.projections.iter();
            let prev_index = loop {
                if let Some(k) = hash_map.get(&lookup_place) {
                    break Some(*k);
                }
                match it.next() {
                    Some(it) => {
                        lookup_place.projections.push(it.clone());
                    }
                    None => break None,
                }
            };
            match prev_index {
                Some(p) => {
                    let prev_projections_len = self.current_captures[p].place.projections.len();
                    self.truncate_capture_spans(&mut item, prev_projections_len);
                    self.current_captures[p].span_stacks.extend(item.span_stacks);
                    let len = self.current_captures[p].place.projections.len();
                    let kind_after_truncate =
                        item.place.capture_kind_of_truncated_place(item.kind, len);
                    self.current_captures[p].kind =
                        cmp::max(kind_after_truncate, self.current_captures[p].kind);
                }
                None => {
                    hash_map.insert(item.place.clone(), self.current_captures.len());
                    self.current_captures.push(item);
                }
            }
        }
    }

    fn consume_with_pat(&mut self, mut place: HirPlace, tgt_pat: PatId) {
        let adjustments_count =
            self.result.pat_adjustments.get(&tgt_pat).map(|it| it.len()).unwrap_or_default();
        place.projections.extend((0..adjustments_count).map(|_| ProjectionElem::Deref));
        self.current_capture_span_stack
            .extend((0..adjustments_count).map(|_| MirSpan::PatId(tgt_pat)));
        'reset_span_stack: {
            match &self.body[tgt_pat] {
                Pat::Missing | Pat::Wild => (),
                Pat::Tuple { args, ellipsis } => {
                    let (al, ar) = args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
                    let field_count = match self.result[tgt_pat].kind(Interner) {
                        TyKind::Tuple(_, s) => s.len(Interner),
                        _ => break 'reset_span_stack,
                    };
                    let fields = 0..field_count;
                    let it = al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                    for (&arg, i) in it {
                        let mut p = place.clone();
                        self.current_capture_span_stack.push(MirSpan::PatId(arg));
                        p.projections.push(ProjectionElem::Field(Either::Right(TupleFieldId {
                            tuple: TupleId(!0), // dummy this, as its unused anyways
                            index: i as u32,
                        })));
                        self.consume_with_pat(p, arg);
                        self.current_capture_span_stack.pop();
                    }
                }
                Pat::Or(pats) => {
                    for pat in pats.iter() {
                        self.consume_with_pat(place.clone(), *pat);
                    }
                }
                Pat::Record { args, .. } => {
                    let Some(variant) = self.result.variant_resolution_for_pat(tgt_pat) else {
                        break 'reset_span_stack;
                    };
                    match variant {
                        VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                            self.consume_place(place)
                        }
                        VariantId::StructId(s) => {
                            let vd = &*self.db.struct_data(s).variant_data;
                            for field_pat in args.iter() {
                                let arg = field_pat.pat;
                                let Some(local_id) = vd.field(&field_pat.name) else {
                                    continue;
                                };
                                let mut p = place.clone();
                                self.current_capture_span_stack.push(MirSpan::PatId(arg));
                                p.projections.push(ProjectionElem::Field(Either::Left(FieldId {
                                    parent: variant,
                                    local_id,
                                })));
                                self.consume_with_pat(p, arg);
                                self.current_capture_span_stack.pop();
                            }
                        }
                    }
                }
                Pat::Range { .. }
                | Pat::Slice { .. }
                | Pat::ConstBlock(_)
                | Pat::Path(_)
                | Pat::Lit(_) => self.consume_place(place),
                &Pat::Bind { id, subpat: _ } => {
                    let mode = self.result.binding_modes[tgt_pat];
                    let capture_kind = match mode {
                        BindingMode::Move => {
                            self.consume_place(place);
                            break 'reset_span_stack;
                        }
                        BindingMode::Ref(Mutability::Not) => BorrowKind::Shared,
                        BindingMode::Ref(Mutability::Mut) => {
                            BorrowKind::Mut { kind: MutBorrowKind::Default }
                        }
                    };
                    self.current_capture_span_stack.push(MirSpan::BindingId(id));
                    self.add_capture(place, CaptureKind::ByRef(capture_kind));
                    self.current_capture_span_stack.pop();
                }
                Pat::TupleStruct { path: _, args, ellipsis } => {
                    let Some(variant) = self.result.variant_resolution_for_pat(tgt_pat) else {
                        break 'reset_span_stack;
                    };
                    match variant {
                        VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                            self.consume_place(place)
                        }
                        VariantId::StructId(s) => {
                            let vd = &*self.db.struct_data(s).variant_data;
                            let (al, ar) =
                                args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
                            let fields = vd.fields().iter();
                            let it = al
                                .iter()
                                .zip(fields.clone())
                                .chain(ar.iter().rev().zip(fields.rev()));
                            for (&arg, (i, _)) in it {
                                let mut p = place.clone();
                                self.current_capture_span_stack.push(MirSpan::PatId(arg));
                                p.projections.push(ProjectionElem::Field(Either::Left(FieldId {
                                    parent: variant,
                                    local_id: i,
                                })));
                                self.consume_with_pat(p, arg);
                                self.current_capture_span_stack.pop();
                            }
                        }
                    }
                }
                Pat::Ref { pat, mutability: _ } => {
                    self.current_capture_span_stack.push(MirSpan::PatId(tgt_pat));
                    place.projections.push(ProjectionElem::Deref);
                    self.consume_with_pat(place, *pat);
                    self.current_capture_span_stack.pop();
                }
                Pat::Box { .. } => (), // not supported
            }
        }
        self.current_capture_span_stack
            .truncate(self.current_capture_span_stack.len() - adjustments_count);
    }

    fn consume_exprs(&mut self, exprs: impl Iterator<Item = ExprId>) {
        for expr in exprs {
            self.consume_expr(expr);
        }
    }

    fn closure_kind(&self) -> FnTrait {
        let mut r = FnTrait::Fn;
        for it in &self.current_captures {
            r = cmp::min(
                r,
                match &it.kind {
                    CaptureKind::ByRef(BorrowKind::Mut { .. }) => FnTrait::FnMut,
                    CaptureKind::ByRef(BorrowKind::Shallow | BorrowKind::Shared) => FnTrait::Fn,
                    CaptureKind::ByValue => FnTrait::FnOnce,
                },
            )
        }
        r
    }

    fn analyze_closure(&mut self, closure: ClosureId) -> FnTrait {
        let InternedClosure(_, root) = self.db.lookup_intern_closure(closure.into());
        self.current_closure = Some(closure);
        let Expr::Closure { body, capture_by, .. } = &self.body[root] else {
            unreachable!("Closure expression id is always closure");
        };
        self.consume_expr(*body);
        for item in &self.current_captures {
            if matches!(
                item.kind,
                CaptureKind::ByRef(BorrowKind::Mut {
                    kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow
                })
            ) && !item.place.projections.contains(&ProjectionElem::Deref)
            {
                // FIXME: remove the `mutated_bindings_in_closure` completely and add proper fake reads in
                // MIR. I didn't do that due duplicate diagnostics.
                self.result.mutated_bindings_in_closure.insert(item.place.local);
            }
        }
        self.restrict_precision_for_unsafe();
        // `closure_kind` should be done before adjust_for_move_closure
        // If there exists pre-deduced kind of a closure, use it instead of one determined by capture, as rustc does.
        // rustc also does diagnostics here if the latter is not a subtype of the former.
        let closure_kind = self
            .result
            .closure_info
            .get(&closure)
            .map_or_else(|| self.closure_kind(), |info| info.1);
        match capture_by {
            CaptureBy::Value => self.adjust_for_move_closure(),
            CaptureBy::Ref => (),
        }
        self.minimize_captures();
        self.strip_captures_ref_span();
        let result = mem::take(&mut self.current_captures);
        let captures = result.into_iter().map(|it| it.with_ty(self)).collect::<Vec<_>>();
        self.result.closure_info.insert(closure, (captures, closure_kind));
        closure_kind
    }

    fn strip_captures_ref_span(&mut self) {
        // FIXME: Borrow checker won't allow without this.
        let mut captures = std::mem::take(&mut self.current_captures);
        for capture in &mut captures {
            if matches!(capture.kind, CaptureKind::ByValue) {
                for span_stack in &mut capture.span_stacks {
                    if span_stack[span_stack.len() - 1].is_ref_span(self.body) {
                        span_stack.truncate(span_stack.len() - 1);
                    }
                }
            }
        }
        self.current_captures = captures;
    }

    pub(crate) fn infer_closures(&mut self) {
        let deferred_closures = self.sort_closures();
        for (closure, exprs) in deferred_closures.into_iter().rev() {
            self.current_captures = vec![];
            let kind = self.analyze_closure(closure);

            for (derefed_callee, callee_ty, params, expr) in exprs {
                if let &Expr::Call { callee, .. } = &self.body[expr] {
                    let mut adjustments =
                        self.result.expr_adjustments.remove(&callee).unwrap_or_default();
                    self.write_fn_trait_method_resolution(
                        kind,
                        &derefed_callee,
                        &mut adjustments,
                        &callee_ty,
                        &params,
                        expr,
                    );
                    self.result.expr_adjustments.insert(callee, adjustments);
                }
            }
        }
    }

    /// We want to analyze some closures before others, to have a correct analysis:
    /// * We should analyze nested closures before the parent, since the parent should capture some of
    ///   the things that its children captures.
    /// * If a closure calls another closure, we need to analyze the callee, to find out how we should
    ///   capture it (e.g. by move for FnOnce)
    ///
    /// These dependencies are collected in the main inference. We do a topological sort in this function. It
    /// will consume the `deferred_closures` field and return its content in a sorted vector.
    fn sort_closures(&mut self) -> Vec<(ClosureId, Vec<(Ty, Ty, Vec<Ty>, ExprId)>)> {
        let mut deferred_closures = mem::take(&mut self.deferred_closures);
        let mut dependents_count: FxHashMap<ClosureId, usize> =
            deferred_closures.keys().map(|it| (*it, 0)).collect();
        for deps in self.closure_dependencies.values() {
            for dep in deps {
                *dependents_count.entry(*dep).or_default() += 1;
            }
        }
        let mut queue: Vec<_> =
            deferred_closures.keys().copied().filter(|it| dependents_count[it] == 0).collect();
        let mut result = vec![];
        while let Some(it) = queue.pop() {
            if let Some(d) = deferred_closures.remove(&it) {
                result.push((it, d));
            }
            for dep in self.closure_dependencies.get(&it).into_iter().flat_map(|it| it.iter()) {
                let cnt = dependents_count.get_mut(dep).unwrap();
                *cnt -= 1;
                if *cnt == 0 {
                    queue.push(*dep);
                }
            }
        }
        result
    }
}

/// Call this only when the last span in the stack isn't a split.
fn apply_adjusts_to_place(
    current_capture_span_stack: &mut Vec<MirSpan>,
    mut r: HirPlace,
    adjustments: &[Adjustment],
) -> Option<HirPlace> {
    let span = *current_capture_span_stack.last().expect("empty capture span stack");
    for adj in adjustments {
        match &adj.kind {
            Adjust::Deref(None) => {
                current_capture_span_stack.push(span);
                r.projections.push(ProjectionElem::Deref);
            }
            _ => return None,
        }
    }
    Some(r)
}
