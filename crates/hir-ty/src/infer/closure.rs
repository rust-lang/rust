//! Inference of closure parameter types based on the closure's expected type.

use std::{cmp, collections::HashMap, convert::Infallible, mem};

use chalk_ir::{cast::Cast, AliasEq, AliasTy, FnSubst, Mutability, TyKind, WhereClause};
use hir_def::{
    hir::{
        Array, BinaryOp, BindingAnnotation, BindingId, CaptureBy, Expr, ExprId, Pat, PatId,
        Statement, UnaryOp,
    },
    lang_item::LangItem,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    FieldId, HasModule, VariantId,
};
use hir_expand::name;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stdx::never;

use crate::{
    mir::{BorrowKind, ProjectionElem},
    static_lifetime, to_chalk_trait_id,
    traits::FnTrait,
    utils::{self, pattern_matching_dereference_count},
    Adjust, Adjustment, Canonical, CanonicalVarKinds, ChalkTraitId, ClosureId, DynTy, FnPointer,
    FnSig, InEnvironment, Interner, Substitution, Ty, TyBuilder, TyExt,
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
            ty = p.projected_ty(ty, ctx.db, |_, _| {
                unreachable!("Closure field only happens in MIR");
            });
        }
        ty.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CaptureKind {
    ByRef(BorrowKind),
    ByValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapturedItem {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    pub(crate) ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapturedItemWithoutTy {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
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
        CapturedItem { place: self.place, kind: self.kind, ty }
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
            _ => (),
        }
        None
    }

    fn push_capture(&mut self, capture: CapturedItemWithoutTy) {
        self.current_captures.push(capture);
    }

    fn ref_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.add_capture(place, CaptureKind::ByRef(BorrowKind::Shared));
        }
        self.walk_expr(expr);
    }

    fn add_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        if self.is_upvar(&place) {
            self.push_capture(CapturedItemWithoutTy { place, kind });
        }
    }

    fn mutate_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { allow_two_phase_borrow: false }),
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
            let ty = place.ty(self).clone();
            let kind = if self.is_ty_copy(ty) {
                CaptureKind::ByRef(BorrowKind::Shared)
            } else {
                CaptureKind::ByValue
            };
            self.push_capture(CapturedItemWithoutTy { place, kind });
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
                if self.is_upvar(&place) {
                    self.push_capture(CapturedItemWithoutTy { place, kind: capture_kind });
                }
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
            | Expr::Const { statements, tail, .. }
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
                                    let ty = self.expr_ty(*initializer);
                                    self.consume_with_pat(
                                        place,
                                        ty,
                                        BindingAnnotation::Unannotated,
                                        *pat,
                                    );
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
            Expr::While { condition, body, label: _ }
            | Expr::For { iterable: condition, pat: _, body, label: _ } => {
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
                self.consume_expr(*expr);
                for arm in arms.iter() {
                    self.consume_expr(arm.expr);
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
                if let Some((f, _)) = self.result.method_resolution(tgt_expr) {
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
                cc.extend(
                    captures
                        .iter()
                        .filter(|x| self.is_upvar(&x.place))
                        .map(|x| CapturedItemWithoutTy { place: x.place.clone(), kind: x.kind }),
                );
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
            | Expr::Underscore => (),
        }
    }

    fn expr_ty(&mut self, expr: ExprId) -> Ty {
        self.result[expr].clone()
    }

    fn is_upvar(&self, place: &HirPlace) -> bool {
        let b = &self.body[place.local];
        if let Some(c) = self.current_closure {
            let (_, root) = self.db.lookup_intern_closure(c.into());
            return b.is_upvar(root);
        }
        false
    }

    fn is_ty_copy(&self, ty: Ty) -> bool {
        if let TyKind::Closure(id, _) = ty.kind(Interner) {
            // FIXME: We handle closure as a special case, since chalk consider every closure as copy. We
            // should probably let chalk know which closures are copy, but I don't know how doing it
            // without creating query cycles.
            return self.result.closure_info.get(id).map(|x| x.1 == FnTrait::Fn).unwrap_or(true);
        }
        let crate_id = self.owner.module(self.db.upcast()).krate();
        let Some(copy_trait) = self.db.lang_item(crate_id, LangItem::Copy).and_then(|x| x.as_trait()) else {
            return false;
        };
        let trait_ref = TyBuilder::trait_ref(self.db, copy_trait).push(ty).build();
        let env = self.db.trait_environment_for_body(self.owner);
        let goal = Canonical {
            value: InEnvironment::new(&env.env, trait_ref.cast(Interner)),
            binders: CanonicalVarKinds::empty(Interner),
        };
        self.db.trait_solve(crate_id, None, goal).is_some()
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
                    self.current_captures[p].kind =
                        cmp::max(item.kind, self.current_captures[p].kind);
                }
                None => {
                    hash_map.insert(item.place.clone(), self.current_captures.len());
                    self.current_captures.push(item);
                }
            }
        }
    }

    fn consume_with_pat(
        &mut self,
        mut place: HirPlace,
        mut ty: Ty,
        mut bm: BindingAnnotation,
        pat: PatId,
    ) {
        match &self.body[pat] {
            Pat::Missing | Pat::Wild => (),
            Pat::Tuple { args, ellipsis } => {
                pattern_matching_dereference(&mut ty, &mut bm, &mut place);
                let (al, ar) = args.split_at(ellipsis.unwrap_or(args.len()));
                let subst = match ty.kind(Interner) {
                    TyKind::Tuple(_, s) => s,
                    _ => return,
                };
                let fields = subst.iter(Interner).map(|x| x.assert_ty_ref(Interner)).enumerate();
                let it = al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                for (arg, (i, ty)) in it {
                    let mut p = place.clone();
                    p.projections.push(ProjectionElem::TupleOrClosureField(i));
                    self.consume_with_pat(p, ty.clone(), bm, *arg);
                }
            }
            Pat::Or(pats) => {
                for pat in pats.iter() {
                    self.consume_with_pat(place.clone(), ty.clone(), bm, *pat);
                }
            }
            Pat::Record { args, .. } => {
                pattern_matching_dereference(&mut ty, &mut bm, &mut place);
                let subst = match ty.kind(Interner) {
                    TyKind::Adt(_, s) => s,
                    _ => return,
                };
                let Some(variant) = self.result.variant_resolution_for_pat(pat) else {
                    return;
                };
                match variant {
                    VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                        self.consume_place(place)
                    }
                    VariantId::StructId(s) => {
                        let vd = &*self.db.struct_data(s).variant_data;
                        let field_types = self.db.field_types(variant);
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
                            self.consume_with_pat(
                                p,
                                field_types[local_id].clone().substitute(Interner, subst),
                                bm,
                                arg,
                            );
                        }
                    }
                }
            }
            Pat::Range { .. }
            | Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Path(_)
            | Pat::Lit(_) => self.consume_place(place),
            Pat::Bind { id, subpat: _ } => {
                let mode = self.body.bindings[*id].mode;
                if matches!(mode, BindingAnnotation::Ref | BindingAnnotation::RefMut) {
                    bm = mode;
                }
                let capture_kind = match bm {
                    BindingAnnotation::Unannotated | BindingAnnotation::Mutable => {
                        self.consume_place(place);
                        return;
                    }
                    BindingAnnotation::Ref => BorrowKind::Shared,
                    BindingAnnotation::RefMut => BorrowKind::Mut { allow_two_phase_borrow: false },
                };
                self.add_capture(place, CaptureKind::ByRef(capture_kind));
            }
            Pat::TupleStruct { path: _, args, ellipsis } => {
                pattern_matching_dereference(&mut ty, &mut bm, &mut place);
                let subst = match ty.kind(Interner) {
                    TyKind::Adt(_, s) => s,
                    _ => return,
                };
                let Some(variant) = self.result.variant_resolution_for_pat(pat) else {
                    return;
                };
                match variant {
                    VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                        self.consume_place(place)
                    }
                    VariantId::StructId(s) => {
                        let vd = &*self.db.struct_data(s).variant_data;
                        let (al, ar) = args.split_at(ellipsis.unwrap_or(args.len()));
                        let fields = vd.fields().iter();
                        let it =
                            al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                        let field_types = self.db.field_types(variant);
                        for (arg, (i, _)) in it {
                            let mut p = place.clone();
                            p.projections.push(ProjectionElem::Field(FieldId {
                                parent: variant.into(),
                                local_id: i,
                            }));
                            self.consume_with_pat(
                                p,
                                field_types[i].clone().substitute(Interner, subst),
                                bm,
                                *arg,
                            );
                        }
                    }
                }
            }
            Pat::Ref { pat, mutability: _ } => {
                if let Some((inner, _, _)) = ty.as_reference() {
                    ty = inner.clone();
                    place.projections.push(ProjectionElem::Deref);
                    self.consume_with_pat(place, ty, bm, *pat)
                }
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
            if matches!(item.kind, CaptureKind::ByRef(BorrowKind::Mut { .. })) {
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

fn pattern_matching_dereference(
    cond_ty: &mut Ty,
    binding_mode: &mut BindingAnnotation,
    cond_place: &mut HirPlace,
) {
    let cnt = pattern_matching_dereference_count(cond_ty, binding_mode);
    cond_place.projections.extend((0..cnt).map(|_| ProjectionElem::Deref));
}
