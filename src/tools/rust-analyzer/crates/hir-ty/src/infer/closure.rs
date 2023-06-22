//! Inference of closure parameter types based on the closure's expected type.

use std::{cmp, collections::HashMap, convert::Infallible, mem};

use chalk_ir::{
    cast::Cast,
    fold::{FallibleTypeFolder, TypeFoldable},
    AliasEq, AliasTy, BoundVar, DebruijnIndex, FnSubst, Mutability, TyKind, WhereClause,
};
use hir_def::{
    data::adt::VariantData,
    hir::{Array, BinaryOp, BindingId, CaptureBy, Expr, ExprId, Pat, PatId, Statement, UnaryOp},
    lang_item::LangItem,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId, FieldId, HasModule, VariantId,
};
use hir_expand::name;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stdx::never;

use crate::{
    db::HirDatabase,
    from_placeholder_idx, make_binders,
    mir::{BorrowKind, MirSpan, ProjectionElem},
    static_lifetime, to_chalk_trait_id,
    traits::FnTrait,
    utils::{self, generics, Generics},
    Adjust, Adjustment, Binders, BindingMode, ChalkTraitId, ClosureId, DynTy, FnPointer, FnSig,
    Interner, Substitution, Ty, TyExt,
};

use super::{Expectation, InferenceContext};

impl InferenceContext<'_> {
    // This function handles both closures and generators.
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

        // Deduction from where-clauses in scope, as well as fn-pointer coercion are handled here.
        let _ = self.coerce(Some(closure_expr), closure_ty, &expected_ty);

        // Generators are not Fn* so return early.
        if matches!(closure_ty.kind(Interner), TyKind::Generator(..)) {
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
                        sig: FnSig { abi: (), safety: chalk_ir::Safety::Safe, variadic: false },
                        substitution: FnSubst(Substitution::from_iter(Interner, sig_tys)),
                    });
                }
            }
        }

        None
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
        ty.clone()
    }

    fn capture_kind_of_truncated_place(
        &self,
        mut current_capture: CaptureKind,
        len: usize,
    ) -> CaptureKind {
        match current_capture {
            CaptureKind::ByRef(BorrowKind::Mut { .. }) => {
                if self.projections[len..].iter().any(|x| *x == ProjectionElem::Deref) {
                    current_capture = CaptureKind::ByRef(BorrowKind::Unique);
                }
            }
            _ => (),
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
    pub(crate) span: MirSpan,
    pub(crate) ty: Binders<Ty>,
}

impl CapturedItem {
    pub fn local(&self) -> BindingId {
        self.place.local
    }

    pub fn ty(&self, subst: &Substitution) -> Ty {
        self.ty.clone().substitute(Interner, utils::ClosureSubst(subst).parent_subst())
    }

    pub fn kind(&self) -> CaptureKind {
        self.kind
    }

    pub fn display_place(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let mut result = body[self.place.local].name.display(db.upcast()).to_string();
        let mut field_need_paren = false;
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {
                    result = format!("*{result}");
                    field_need_paren = true;
                }
                ProjectionElem::Field(f) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    let variant_data = f.parent.variant_data(db.upcast());
                    let field = match &*variant_data {
                        VariantData::Record(fields) => fields[f.local_id]
                            .name
                            .as_str()
                            .unwrap_or("[missing field]")
                            .to_string(),
                        VariantData::Tuple(fields) => fields
                            .iter()
                            .position(|x| x.0 == f.local_id)
                            .unwrap_or_default()
                            .to_string(),
                        VariantData::Unit => "[missing field]".to_string(),
                    };
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                &ProjectionElem::TupleOrClosureField(field) => {
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
    pub(crate) span: MirSpan,
}

impl CapturedItemWithoutTy {
    fn with_ty(self, ctx: &mut InferenceContext<'_>) -> CapturedItem {
        let ty = self.place.ty(ctx).clone();
        let ty = match &self.kind {
            CaptureKind::ByValue => ty,
            CaptureKind::ByRef(bk) => {
                let m = match bk {
                    BorrowKind::Mut { .. } => Mutability::Mut,
                    _ => Mutability::Not,
                };
                TyKind::Ref(m, static_lifetime(), ty).intern(Interner)
            }
        };
        return CapturedItem {
            place: self.place,
            kind: self.kind,
            span: self.span,
            ty: replace_placeholder_with_binder(ctx.db, ctx.owner, ty),
        };

        fn replace_placeholder_with_binder(
            db: &dyn HirDatabase,
            owner: DefWithBodyId,
            ty: Ty,
        ) -> Binders<Ty> {
            struct Filler<'a> {
                db: &'a dyn HirDatabase,
                generics: Generics,
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
                    let Some(idx) = self.generics.param_idx(x) else {
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
                    let Some(idx) = self.generics.param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_ty(Interner))
                }
            }
            let Some(generic_def) = owner.as_generic_def_id() else {
                return Binders::empty(Interner, ty);
            };
            let filler = &mut Filler { db, generics: generics(db.upcast(), generic_def) };
            let result = ty.clone().try_fold_with(filler, DebruijnIndex::INNERMOST).unwrap_or(ty);
            make_binders(db, &filler.generics, result)
        }
    }
}

impl InferenceContext<'_> {
    fn place_of_expr(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        let r = self.place_of_expr_without_adjust(tgt_expr)?;
        let default = vec![];
        let adjustments = self.result.expr_adjustments.get(&tgt_expr).unwrap_or(&default);
        apply_adjusts_to_place(r, adjustments)
    }

    fn place_of_expr_without_adjust(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        match &self.body[tgt_expr] {
            Expr::Path(p) => {
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, tgt_expr);
                if let Some(r) = resolver.resolve_path_in_value_ns(self.db.upcast(), p) {
                    if let ResolveValueResult::ValueNs(v) = r {
                        if let ValueNs::LocalBinding(b) = v {
                            return Some(HirPlace { local: b, projections: vec![] });
                        }
                    }
                }
            }
            Expr::Field { expr, name } => {
                let mut place = self.place_of_expr(*expr)?;
                if let TyKind::Tuple(..) = self.expr_ty(*expr).kind(Interner) {
                    let index = name.as_tuple_index()?;
                    place.projections.push(ProjectionElem::TupleOrClosureField(index))
                } else {
                    let field = self.result.field_resolution(tgt_expr)?;
                    place.projections.push(ProjectionElem::Field(field));
                }
                return Some(place);
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if matches!(
                    self.expr_ty_after_adjustments(*expr).kind(Interner),
                    TyKind::Ref(..) | TyKind::Raw(..)
                ) {
                    let mut place = self.place_of_expr(*expr)?;
                    place.projections.push(ProjectionElem::Deref);
                    return Some(place);
                }
            }
            _ => (),
        }
        None
    }

    fn push_capture(&mut self, capture: CapturedItemWithoutTy) {
        self.current_captures.push(capture);
    }

    fn ref_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.add_capture(place, CaptureKind::ByRef(BorrowKind::Shared), expr.into());
        }
        self.walk_expr(expr);
    }

    fn add_capture(&mut self, place: HirPlace, kind: CaptureKind, span: MirSpan) {
        if self.is_upvar(&place) {
            self.push_capture(CapturedItemWithoutTy { place, kind, span });
        }
    }

    fn mutate_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { allow_two_phase_borrow: false }),
                expr.into(),
            );
        }
        self.walk_expr(expr);
    }

    fn consume_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.consume_place(place, expr.into());
        }
        self.walk_expr(expr);
    }

    fn consume_place(&mut self, place: HirPlace, span: MirSpan) {
        if self.is_upvar(&place) {
            let ty = place.ty(self).clone();
            let kind = if self.is_ty_copy(ty) {
                CaptureKind::ByRef(BorrowKind::Shared)
            } else {
                CaptureKind::ByValue
            };
            self.push_capture(CapturedItemWithoutTy { place, kind, span });
        }
    }

    fn walk_expr_with_adjust(&mut self, tgt_expr: ExprId, adjustment: &[Adjustment]) {
        if let Some((last, rest)) = adjustment.split_last() {
            match last.kind {
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
            Mutability::Mut => {
                CaptureKind::ByRef(BorrowKind::Mut { allow_two_phase_borrow: false })
            }
            Mutability::Not => CaptureKind::ByRef(BorrowKind::Shared),
        };
        if let Some(place) = self.place_of_expr_without_adjust(tgt_expr) {
            if let Some(place) = apply_adjusts_to_place(place, rest) {
                self.add_capture(place, capture_kind, tgt_expr.into());
            }
        }
        self.walk_expr_with_adjust(tgt_expr, rest);
    }

    fn walk_expr(&mut self, tgt_expr: ExprId) {
        if let Some(x) = self.result.expr_adjustments.get_mut(&tgt_expr) {
            // FIXME: this take is completely unneeded, and just is here to make borrow checker
            // happy. Remove it if you can.
            let x_taken = mem::take(x);
            self.walk_expr_with_adjust(tgt_expr, &x_taken);
            *self.result.expr_adjustments.get_mut(&tgt_expr).unwrap() = x_taken;
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn walk_expr_without_adjust(&mut self, tgt_expr: ExprId) {
        match &self.body[tgt_expr] {
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
                                if let Some(initializer) = initializer {
                                    self.consume_expr(*initializer);
                                }
                                return;
                            }
                            if let Some(initializer) = initializer {
                                self.walk_expr(*initializer);
                                if let Some(place) = self.place_of_expr(*initializer) {
                                    self.consume_with_pat(place, *pat);
                                }
                            }
                        }
                        Statement::Expr { expr, has_semi: _ } => {
                            self.consume_expr(*expr);
                        }
                    }
                }
                if let Some(tail) = tail {
                    self.consume_expr(*tail);
                }
            }
            Expr::While { condition, body, label: _ } => {
                self.consume_expr(*condition);
                self.consume_expr(*body);
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
                            self.push_capture(CapturedItemWithoutTy {
                                place: discr_place,
                                kind: c,
                                span: (*expr).into(),
                            })
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
            Expr::RecordLit { fields, spread, .. } => {
                if let &Some(expr) = spread {
                    self.consume_expr(expr);
                }
                self.consume_exprs(fields.iter().map(|x| x.expr));
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
                            self.resolve_lang_item(LangItem::DerefMut).and_then(|x| x.as_trait())
                        {
                            if let Some(deref_fn) =
                                self.db.trait_data(deref_trait).method_by_name(&name![deref_mut])
                            {
                                break 'b deref_fn == f;
                            }
                        }
                        false
                    };
                    if mutability {
                        self.mutate_expr(*expr);
                    } else {
                        self.ref_expr(*expr);
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
            Expr::Ref { expr, rawness: _, mutability } => match mutability {
                hir_def::type_ref::Mutability::Shared => self.ref_expr(*expr),
                hir_def::type_ref::Mutability::Mut => self.mutate_expr(*expr),
            },
            Expr::BinaryOp { lhs, rhs, op } => {
                let Some(op) = op else {
                    return;
                };
                if matches!(op, BinaryOp::Assignment { .. }) {
                    self.mutate_expr(*lhs);
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
            Expr::Index { base, index } => {
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
                cc.extend(captures.iter().filter(|x| self.is_upvar(&x.place)).map(|x| {
                    CapturedItemWithoutTy { place: x.place.clone(), kind: x.kind, span: x.span }
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
            BorrowKind::Mut { allow_two_phase_borrow: false },
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
                    let adt = variant.adt_id();
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
            Pat::Bind { id, .. } => match self.result.binding_modes[*id] {
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
        if self.result.pat_adjustments.get(&p).map_or(false, |x| !x.is_empty()) {
            for_mut = BorrowKind::Unique;
        }
        self.body.walk_pats_shallow(p, |p| self.walk_pat_inner(p, update_result, for_mut));
    }

    fn expr_ty(&self, expr: ExprId) -> Ty {
        self.result[expr].clone()
    }

    fn expr_ty_after_adjustments(&self, e: ExprId) -> Ty {
        let mut ty = None;
        if let Some(x) = self.result.expr_adjustments.get(&e) {
            if let Some(x) = x.last() {
                ty = Some(x.target.clone());
            }
        }
        ty.unwrap_or_else(|| self.expr_ty(e))
    }

    fn is_upvar(&self, place: &HirPlace) -> bool {
        if let Some(c) = self.current_closure {
            let (_, root) = self.db.lookup_intern_closure(c.into());
            return self.body.is_binding_upvar(place.local, root);
        }
        false
    }

    fn is_ty_copy(&mut self, ty: Ty) -> bool {
        if let TyKind::Closure(id, _) = ty.kind(Interner) {
            // FIXME: We handle closure as a special case, since chalk consider every closure as copy. We
            // should probably let chalk know which closures are copy, but I don't know how doing it
            // without creating query cycles.
            return self.result.closure_info.get(id).map(|x| x.1 == FnTrait::Fn).unwrap_or(true);
        }
        self.table.resolve_completely(ty).is_copy(self.db, self.owner)
    }

    fn select_from_expr(&mut self, expr: ExprId) {
        self.walk_expr(expr);
    }

    fn adjust_for_move_closure(&mut self) {
        for capture in &mut self.current_captures {
            if let Some(first_deref) =
                capture.place.projections.iter().position(|proj| *proj == ProjectionElem::Deref)
            {
                capture.place.projections.truncate(first_deref);
            }
            capture.kind = CaptureKind::ByValue;
        }
    }

    fn minimize_captures(&mut self) {
        self.current_captures.sort_by_key(|x| x.place.projections.len());
        let mut hash_map = HashMap::<HirPlace, usize>::new();
        let result = mem::take(&mut self.current_captures);
        for item in result {
            let mut lookup_place = HirPlace { local: item.place.local, projections: vec![] };
            let mut it = item.place.projections.iter();
            let prev_index = loop {
                if let Some(k) = hash_map.get(&lookup_place) {
                    break Some(*k);
                }
                match it.next() {
                    Some(x) => lookup_place.projections.push(x.clone()),
                    None => break None,
                }
            };
            match prev_index {
                Some(p) => {
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

    fn consume_with_pat(&mut self, mut place: HirPlace, pat: PatId) {
        let cnt = self.result.pat_adjustments.get(&pat).map(|x| x.len()).unwrap_or_default();
        place.projections = place
            .projections
            .iter()
            .cloned()
            .chain((0..cnt).map(|_| ProjectionElem::Deref))
            .collect::<Vec<_>>()
            .into();
        match &self.body[pat] {
            Pat::Missing | Pat::Wild => (),
            Pat::Tuple { args, ellipsis } => {
                let (al, ar) = args.split_at(ellipsis.unwrap_or(args.len()));
                let field_count = match self.result[pat].kind(Interner) {
                    TyKind::Tuple(_, s) => s.len(Interner),
                    _ => return,
                };
                let fields = 0..field_count;
                let it = al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                for (arg, i) in it {
                    let mut p = place.clone();
                    p.projections.push(ProjectionElem::TupleOrClosureField(i));
                    self.consume_with_pat(p, *arg);
                }
            }
            Pat::Or(pats) => {
                for pat in pats.iter() {
                    self.consume_with_pat(place.clone(), *pat);
                }
            }
            Pat::Record { args, .. } => {
                let Some(variant) = self.result.variant_resolution_for_pat(pat) else {
                    return;
                };
                match variant {
                    VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                        self.consume_place(place, pat.into())
                    }
                    VariantId::StructId(s) => {
                        let vd = &*self.db.struct_data(s).variant_data;
                        for field_pat in args.iter() {
                            let arg = field_pat.pat;
                            let Some(local_id) = vd.field(&field_pat.name) else {
                                continue;
                            };
                            let mut p = place.clone();
                            p.projections.push(ProjectionElem::Field(FieldId {
                                parent: variant.into(),
                                local_id,
                            }));
                            self.consume_with_pat(p, arg);
                        }
                    }
                }
            }
            Pat::Range { .. }
            | Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Path(_)
            | Pat::Lit(_) => self.consume_place(place, pat.into()),
            Pat::Bind { id, subpat: _ } => {
                let mode = self.result.binding_modes[*id];
                let capture_kind = match mode {
                    BindingMode::Move => {
                        self.consume_place(place, pat.into());
                        return;
                    }
                    BindingMode::Ref(Mutability::Not) => BorrowKind::Shared,
                    BindingMode::Ref(Mutability::Mut) => {
                        BorrowKind::Mut { allow_two_phase_borrow: false }
                    }
                };
                self.add_capture(place, CaptureKind::ByRef(capture_kind), pat.into());
            }
            Pat::TupleStruct { path: _, args, ellipsis } => {
                let Some(variant) = self.result.variant_resolution_for_pat(pat) else {
                    return;
                };
                match variant {
                    VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                        self.consume_place(place, pat.into())
                    }
                    VariantId::StructId(s) => {
                        let vd = &*self.db.struct_data(s).variant_data;
                        let (al, ar) = args.split_at(ellipsis.unwrap_or(args.len()));
                        let fields = vd.fields().iter();
                        let it =
                            al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                        for (arg, (i, _)) in it {
                            let mut p = place.clone();
                            p.projections.push(ProjectionElem::Field(FieldId {
                                parent: variant.into(),
                                local_id: i,
                            }));
                            self.consume_with_pat(p, *arg);
                        }
                    }
                }
            }
            Pat::Ref { pat, mutability: _ } => {
                place.projections.push(ProjectionElem::Deref);
                self.consume_with_pat(place, *pat)
            }
            Pat::Box { .. } => (), // not supported
        }
    }

    fn consume_exprs(&mut self, exprs: impl Iterator<Item = ExprId>) {
        for expr in exprs {
            self.consume_expr(expr);
        }
    }

    fn closure_kind(&self) -> FnTrait {
        let mut r = FnTrait::Fn;
        for x in &self.current_captures {
            r = cmp::min(
                r,
                match &x.kind {
                    CaptureKind::ByRef(BorrowKind::Unique | BorrowKind::Mut { .. }) => {
                        FnTrait::FnMut
                    }
                    CaptureKind::ByRef(BorrowKind::Shallow | BorrowKind::Shared) => FnTrait::Fn,
                    CaptureKind::ByValue => FnTrait::FnOnce,
                },
            )
        }
        r
    }

    fn analyze_closure(&mut self, closure: ClosureId) -> FnTrait {
        let (_, root) = self.db.lookup_intern_closure(closure.into());
        self.current_closure = Some(closure);
        let Expr::Closure { body, capture_by, .. } = &self.body[root] else {
            unreachable!("Closure expression id is always closure");
        };
        self.consume_expr(*body);
        for item in &self.current_captures {
            if matches!(item.kind, CaptureKind::ByRef(BorrowKind::Mut { .. }))
                && !item.place.projections.contains(&ProjectionElem::Deref)
            {
                // FIXME: remove the `mutated_bindings_in_closure` completely and add proper fake reads in
                // MIR. I didn't do that due duplicate diagnostics.
                self.result.mutated_bindings_in_closure.insert(item.place.local);
            }
        }
        // closure_kind should be done before adjust_for_move_closure
        let closure_kind = self.closure_kind();
        match capture_by {
            CaptureBy::Value => self.adjust_for_move_closure(),
            CaptureBy::Ref => (),
        }
        self.minimize_captures();
        let result = mem::take(&mut self.current_captures);
        let captures = result.into_iter().map(|x| x.with_ty(self)).collect::<Vec<_>>();
        self.result.closure_info.insert(closure, (captures, closure_kind));
        closure_kind
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
            deferred_closures.keys().map(|x| (*x, 0)).collect();
        for (_, deps) in &self.closure_dependencies {
            for dep in deps {
                *dependents_count.entry(*dep).or_default() += 1;
            }
        }
        let mut queue: Vec<_> =
            deferred_closures.keys().copied().filter(|x| dependents_count[x] == 0).collect();
        let mut result = vec![];
        while let Some(x) = queue.pop() {
            if let Some(d) = deferred_closures.remove(&x) {
                result.push((x, d));
            }
            for dep in self.closure_dependencies.get(&x).into_iter().flat_map(|x| x.iter()) {
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

fn apply_adjusts_to_place(mut r: HirPlace, adjustments: &[Adjustment]) -> Option<HirPlace> {
    for adj in adjustments {
        match &adj.kind {
            Adjust::Deref(None) => {
                r.projections.push(ProjectionElem::Deref);
            }
            _ => return None,
        }
    }
    Some(r)
}
