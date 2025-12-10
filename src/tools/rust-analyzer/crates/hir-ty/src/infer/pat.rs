//! Type inference for patterns.

use std::{cmp, iter};

use hir_def::{
    HasModule,
    expr_store::{Body, path::Path},
    hir::{Binding, BindingAnnotation, BindingId, Expr, ExprId, Literal, Pat, PatId},
};
use hir_expand::name::Name;
use rustc_ast_ir::Mutability;
use rustc_type_ir::inherent::{GenericArg as _, GenericArgs as _, IntoKind, SliceLike, Ty as _};
use stdx::TupleExt;

use crate::{
    DeclContext, DeclOrigin, InferenceDiagnostic,
    consteval::{self, try_const_usize, usize_const},
    infer::{
        AllowTwoPhase, BindingMode, Expectation, InferenceContext, TypeMismatch, expr::ExprIsRead,
    },
    lower::lower_mutability,
    next_solver::{GenericArgs, Ty, TyKind, Tys, infer::traits::ObligationCause},
};

impl<'db> InferenceContext<'_, 'db> {
    /// Infers type for tuple struct pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_struct_pat_like(
        &mut self,
        path: Option<&Path>,
        expected: Ty<'db>,
        default_bm: BindingMode,
        id: PatId,
        ellipsis: Option<u32>,
        subs: &[PatId],
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let (ty, def) = self.resolve_variant(id.into(), path, true);
        let var_data = def.map(|it| it.fields(self.db));
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }
        if let Some(var) = &var_data {
            let cmp = if ellipsis.is_some() { usize::gt } else { usize::ne };

            if cmp(&subs.len(), &var.fields().len()) {
                self.push_diagnostic(InferenceDiagnostic::MismatchedTupleStructPatArgCount {
                    pat: id.into(),
                    expected: var.fields().len(),
                    found: subs.len(),
                });
            }
        }

        self.unify(ty, expected);

        match def {
            _ if subs.is_empty() => {}
            Some(def) => {
                let field_types = self.db.field_types(def);
                let variant_data = def.fields(self.db);
                let visibilities = self.db.field_visibilities(def);

                let (pre, post) = match ellipsis {
                    Some(idx) => subs.split_at(idx as usize),
                    None => (subs, &[][..]),
                };
                let post_idx_offset = field_types.iter().count().saturating_sub(post.len());

                let pre_iter = pre.iter().enumerate();
                let post_iter = (post_idx_offset..).zip(post.iter());

                let substs = ty.as_adt().map(TupleExt::tail);

                for (i, &subpat) in pre_iter.chain(post_iter) {
                    let expected_ty = {
                        match variant_data.field(&Name::new_tuple_field(i)) {
                            Some(local_id) => {
                                if !visibilities[local_id]
                                    .is_visible_from(self.db, self.resolver.module())
                                {
                                    // FIXME(DIAGNOSE): private tuple field
                                }
                                let f = field_types[local_id];
                                let expected_ty = match substs {
                                    Some(substs) => f.instantiate(self.interner(), substs),
                                    None => f.instantiate(self.interner(), &[]),
                                };
                                self.process_remote_user_written_ty(expected_ty)
                            }
                            None => self.err_ty(),
                        }
                    };

                    self.infer_pat(subpat, expected_ty, default_bm, decl);
                }
            }
            None => {
                let err_ty = self.err_ty();
                for &inner in subs {
                    self.infer_pat(inner, err_ty, default_bm, decl);
                }
            }
        }

        ty
    }

    /// Infers type for record pattern or its corresponding assignee expression.
    pub(super) fn infer_record_pat_like(
        &mut self,
        path: Option<&Path>,
        expected: Ty<'db>,
        default_bm: BindingMode,
        id: PatId,
        subs: impl ExactSizeIterator<Item = (Name, PatId)>,
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let (ty, def) = self.resolve_variant(id.into(), path, false);
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }

        self.unify(ty, expected);

        match def {
            _ if subs.len() == 0 => {}
            Some(def) => {
                let field_types = self.db.field_types(def);
                let variant_data = def.fields(self.db);
                let visibilities = self.db.field_visibilities(def);

                let substs = ty.as_adt().map(TupleExt::tail);

                for (name, inner) in subs {
                    let expected_ty = {
                        match variant_data.field(&name) {
                            Some(local_id) => {
                                if !visibilities[local_id]
                                    .is_visible_from(self.db, self.resolver.module())
                                {
                                    self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                        field: inner.into(),
                                        private: Some(local_id),
                                        variant: def,
                                    });
                                }
                                let f = field_types[local_id];
                                let expected_ty = match substs {
                                    Some(substs) => f.instantiate(self.interner(), substs),
                                    None => f.instantiate(self.interner(), &[]),
                                };
                                self.process_remote_user_written_ty(expected_ty)
                            }
                            None => {
                                self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                    field: inner.into(),
                                    private: None,
                                    variant: def,
                                });
                                self.err_ty()
                            }
                        }
                    };

                    self.infer_pat(inner, expected_ty, default_bm, decl);
                }
            }
            None => {
                let err_ty = self.err_ty();
                for (_, inner) in subs {
                    self.infer_pat(inner, err_ty, default_bm, decl);
                }
            }
        }

        ty
    }

    /// Infers type for tuple pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_pat_like(
        &mut self,
        pat: PatId,
        expected: Ty<'db>,
        default_bm: BindingMode,
        ellipsis: Option<u32>,
        elements: &[PatId],
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let mut expected_len = elements.len();
        if ellipsis.is_some() {
            // Require known type only when `..` is present.
            if let TyKind::Tuple(tys) = self.table.structurally_resolve_type(expected).kind() {
                expected_len = tys.len();
            }
        }
        let max_len = cmp::max(expected_len, elements.len());

        let element_tys_iter = (0..max_len).map(|_| self.table.next_ty_var());
        let element_tys = Tys::new_from_iter(self.interner(), element_tys_iter);
        let pat_ty = Ty::new(self.interner(), TyKind::Tuple(element_tys));
        if self.demand_eqtype(pat.into(), expected, pat_ty).is_err()
            && let TyKind::Tuple(expected) = expected.kind()
        {
            // Equate expected type with the infer vars, for better diagnostics.
            for (expected, elem_ty) in iter::zip(expected, element_tys) {
                _ = self
                    .table
                    .at(&ObligationCause::dummy())
                    .eq(expected, elem_ty)
                    .map(|infer_ok| self.table.register_infer_ok(infer_ok));
            }
        }
        let (before_ellipsis, after_ellipsis) = match ellipsis {
            Some(ellipsis) => {
                let element_tys = element_tys.as_slice();
                // Don't check patterns twice.
                let from_end_start = cmp::max(
                    element_tys.len().saturating_sub(elements.len() - ellipsis as usize),
                    ellipsis as usize,
                );
                (
                    element_tys.get(..ellipsis as usize).unwrap_or(element_tys),
                    element_tys.get(from_end_start..).unwrap_or_default(),
                )
            }
            None => (element_tys.as_slice(), &[][..]),
        };
        for (&elem, &elem_ty) in iter::zip(elements, before_ellipsis.iter().chain(after_ellipsis)) {
            self.infer_pat(elem, elem_ty, default_bm, decl);
        }
        if let Some(uncovered) = elements.get(element_tys.len()..) {
            for &elem in uncovered {
                self.infer_pat(elem, self.types.error, default_bm, decl);
            }
        }
        pat_ty
    }

    /// The resolver needs to be updated to the surrounding expression when inside assignment
    /// (because there, `Pat::Path` can refer to a variable).
    pub(super) fn infer_top_pat(
        &mut self,
        pat: PatId,
        expected: Ty<'db>,
        decl: Option<DeclContext>,
    ) {
        self.infer_pat(pat, expected, BindingMode::default(), decl);
    }

    fn infer_pat(
        &mut self,
        pat: PatId,
        expected: Ty<'db>,
        mut default_bm: BindingMode,
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let mut expected = self.table.structurally_resolve_type(expected);

        if matches!(&self.body[pat], Pat::Ref { .. }) || self.inside_assignment {
            cov_mark::hit!(match_ergonomics_ref);
            // When you encounter a `&pat` pattern, reset to Move.
            // This is so that `w` is by value: `let (_, &w) = &(1, &2);`
            // Destructuring assignments also reset the binding mode and
            // don't do match ergonomics.
            default_bm = BindingMode::Move;
        } else if self.is_non_ref_pat(self.body, pat) {
            let mut pat_adjustments = Vec::new();
            while let TyKind::Ref(_lifetime, inner, mutability) = expected.kind() {
                pat_adjustments.push(expected);
                expected = self.table.try_structurally_resolve_type(inner);
                default_bm = match default_bm {
                    BindingMode::Move => BindingMode::Ref(mutability),
                    BindingMode::Ref(Mutability::Not) => BindingMode::Ref(Mutability::Not),
                    BindingMode::Ref(Mutability::Mut) => BindingMode::Ref(mutability),
                }
            }

            if !pat_adjustments.is_empty() {
                pat_adjustments.shrink_to_fit();
                self.result.pat_adjustments.insert(pat, pat_adjustments);
            }
        }

        // Lose mutability.
        let default_bm = default_bm;
        let expected = expected;

        let ty = match &self.body[pat] {
            Pat::Tuple { args, ellipsis } => {
                self.infer_tuple_pat_like(pat, expected, default_bm, *ellipsis, args, decl)
            }
            Pat::Or(pats) => {
                for pat in pats.iter() {
                    self.infer_pat(*pat, expected, default_bm, decl);
                }
                expected
            }
            &Pat::Ref { pat, mutability } => {
                self.infer_ref_pat(pat, lower_mutability(mutability), expected, default_bm, decl)
            }
            Pat::TupleStruct { path: p, args: subpats, ellipsis } => self
                .infer_tuple_struct_pat_like(
                    p.as_deref(),
                    expected,
                    default_bm,
                    pat,
                    *ellipsis,
                    subpats,
                    decl,
                ),
            Pat::Record { path: p, args: fields, ellipsis: _ } => {
                let subs = fields.iter().map(|f| (f.name.clone(), f.pat));
                self.infer_record_pat_like(p.as_deref(), expected, default_bm, pat, subs, decl)
            }
            Pat::Path(path) => {
                let ty = self.infer_path(path, pat.into()).unwrap_or_else(|| self.err_ty());
                let ty_inserted_vars = self.insert_type_vars_shallow(ty);
                match self.coerce(
                    pat.into(),
                    expected,
                    ty_inserted_vars,
                    AllowTwoPhase::No,
                    ExprIsRead::No,
                ) {
                    Ok(coerced_ty) => {
                        self.write_pat_ty(pat, coerced_ty);
                        return self.pat_ty_after_adjustment(pat);
                    }
                    Err(_) => {
                        self.result.type_mismatches.get_or_insert_default().insert(
                            pat.into(),
                            TypeMismatch { expected, actual: ty_inserted_vars },
                        );
                        self.write_pat_ty(pat, ty);
                        // We return `expected` to prevent cascading errors. I guess an alternative is to
                        // not emit type mismatches for error types and emit an error type here.
                        return expected;
                    }
                }
            }
            Pat::Bind { id, subpat } => {
                return self.infer_bind_pat(pat, *id, default_bm, *subpat, expected, decl);
            }
            Pat::Slice { prefix, slice, suffix } => {
                self.infer_slice_pat(expected, prefix, *slice, suffix, default_bm, decl)
            }
            Pat::Wild => expected,
            Pat::Range { start, end, range_type: _ } => {
                if let Some(start) = *start {
                    let start_ty = self.infer_expr(start, &Expectation::None, ExprIsRead::Yes);
                    _ = self.demand_eqtype(start.into(), expected, start_ty);
                }
                if let Some(end) = *end {
                    let end_ty = self.infer_expr(end, &Expectation::None, ExprIsRead::Yes);
                    _ = self.demand_eqtype(end.into(), expected, end_ty);
                }
                expected
            }
            &Pat::Lit(expr) => {
                // Don't emit type mismatches again, the expression lowering already did that.
                let ty = self.infer_lit_pat(expr, expected);
                self.write_pat_ty(pat, ty);
                return self.pat_ty_after_adjustment(pat);
            }
            Pat::Box { inner } => match self.resolve_boxed_box() {
                Some(box_adt) => {
                    let (inner_ty, alloc_ty) = match expected.as_adt() {
                        Some((adt, subst)) if adt == box_adt => {
                            (subst.type_at(0), subst.as_slice().get(1).and_then(|a| a.as_type()))
                        }
                        _ => (self.types.error, None),
                    };

                    let inner_ty = self.infer_pat(*inner, inner_ty, default_bm, decl);
                    Ty::new_adt(
                        self.interner(),
                        box_adt,
                        GenericArgs::fill_with_defaults(
                            self.interner(),
                            box_adt.into(),
                            iter::once(inner_ty.into()).chain(alloc_ty.map(Into::into)),
                            |_, id, _| self.table.next_var_for_param(id),
                        ),
                    )
                }
                None => self.err_ty(),
            },
            Pat::ConstBlock(expr) => {
                let old_inside_assign = std::mem::replace(&mut self.inside_assignment, false);
                let result =
                    self.infer_expr(*expr, &Expectation::has_type(expected), ExprIsRead::Yes);
                self.inside_assignment = old_inside_assign;
                result
            }
            Pat::Expr(expr) => {
                let old_inside_assign = std::mem::replace(&mut self.inside_assignment, false);
                // LHS of assignment doesn't constitute reads.
                let expr_is_read = ExprIsRead::No;
                let result =
                    self.infer_expr_coerce(*expr, &Expectation::has_type(expected), expr_is_read);
                // We are returning early to avoid the unifiability check below.
                let lhs_ty = self.insert_type_vars_shallow(result);
                let ty = match self.coerce(
                    (*expr).into(),
                    expected,
                    lhs_ty,
                    AllowTwoPhase::No,
                    expr_is_read,
                ) {
                    Ok(ty) => ty,
                    Err(_) => {
                        self.result
                            .type_mismatches
                            .get_or_insert_default()
                            .insert(pat.into(), TypeMismatch { expected, actual: lhs_ty });
                        // `rhs_ty` is returned so no further type mismatches are
                        // reported because of this mismatch.
                        expected
                    }
                };
                self.write_pat_ty(pat, ty);
                self.inside_assignment = old_inside_assign;
                return ty;
            }
            Pat::Missing => self.err_ty(),
        };
        // use a new type variable if we got error type here
        let ty = self.insert_type_vars_shallow(ty);
        // FIXME: This never check is odd, but required with out we do inference right now
        if !expected.is_never() && !self.unify(ty, expected) {
            self.result
                .type_mismatches
                .get_or_insert_default()
                .insert(pat.into(), TypeMismatch { expected, actual: ty });
        }
        self.write_pat_ty(pat, ty);
        self.pat_ty_after_adjustment(pat)
    }

    fn pat_ty_after_adjustment(&self, pat: PatId) -> Ty<'db> {
        *self
            .result
            .pat_adjustments
            .get(&pat)
            .and_then(|it| it.last())
            .unwrap_or(&self.result.type_of_pat[pat])
    }

    fn infer_ref_pat(
        &mut self,
        inner_pat: PatId,
        mutability: Mutability,
        expected: Ty<'db>,
        default_bm: BindingMode,
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let (expectation_type, expectation_lt) = match expected.kind() {
            TyKind::Ref(lifetime, inner_ty, _exp_mut) => (inner_ty, lifetime),
            _ => {
                let inner_ty = self.table.next_ty_var();
                let inner_lt = self.table.next_region_var();
                let ref_ty = Ty::new_ref(self.interner(), inner_lt, inner_ty, mutability);
                // Unification failure will be reported by the caller.
                self.unify(ref_ty, expected);
                (inner_ty, inner_lt)
            }
        };
        let subty = self.infer_pat(inner_pat, expectation_type, default_bm, decl);
        Ty::new_ref(self.interner(), expectation_lt, subty, mutability)
    }

    fn infer_bind_pat(
        &mut self,
        pat: PatId,
        binding: BindingId,
        default_bm: BindingMode,
        subpat: Option<PatId>,
        expected: Ty<'db>,
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let Binding { mode, .. } = self.body[binding];
        let mode = if mode == BindingAnnotation::Unannotated {
            default_bm
        } else {
            BindingMode::convert(mode)
        };
        self.result.binding_modes.insert(pat, mode);

        let inner_ty = match subpat {
            Some(subpat) => self.infer_pat(subpat, expected, default_bm, decl),
            None => expected,
        };
        let inner_ty = self.insert_type_vars_shallow(inner_ty);

        let bound_ty = match mode {
            BindingMode::Ref(mutability) => {
                let inner_lt = self.table.next_region_var();
                Ty::new_ref(self.interner(), inner_lt, expected, mutability)
            }
            BindingMode::Move => expected,
        };
        self.write_pat_ty(pat, inner_ty);
        self.write_binding_ty(binding, bound_ty);
        inner_ty
    }

    fn infer_slice_pat(
        &mut self,
        expected: Ty<'db>,
        prefix: &[PatId],
        slice: Option<PatId>,
        suffix: &[PatId],
        default_bm: BindingMode,
        decl: Option<DeclContext>,
    ) -> Ty<'db> {
        let expected = self.table.structurally_resolve_type(expected);

        // If `expected` is an infer ty, we try to equate it to an array if the given pattern
        // allows it. See issue #16609
        if self.pat_is_irrefutable(decl)
            && expected.is_ty_var()
            && let Some(resolved_array_ty) =
                self.try_resolve_slice_ty_to_array_ty(prefix, suffix, slice)
        {
            self.unify(expected, resolved_array_ty);
        }

        let expected = self.table.try_structurally_resolve_type(expected);
        let elem_ty = match expected.kind() {
            TyKind::Array(st, _) | TyKind::Slice(st) => st,
            _ => self.err_ty(),
        };

        for &pat_id in prefix.iter().chain(suffix.iter()) {
            self.infer_pat(pat_id, elem_ty, default_bm, decl);
        }

        if let Some(slice_pat_id) = slice {
            let rest_pat_ty = match expected.kind() {
                TyKind::Array(_, length) => {
                    let len = try_const_usize(self.db, length);
                    let len =
                        len.and_then(|len| len.checked_sub((prefix.len() + suffix.len()) as u128));
                    Ty::new_array_with_const_len(
                        self.interner(),
                        elem_ty,
                        usize_const(self.db, len, self.resolver.krate()),
                    )
                }
                _ => Ty::new_slice(self.interner(), elem_ty),
            };
            self.infer_pat(slice_pat_id, rest_pat_ty, default_bm, decl);
        }

        match expected.kind() {
            TyKind::Array(_, const_) => {
                Ty::new_array_with_const_len(self.interner(), elem_ty, const_)
            }
            _ => Ty::new_slice(self.interner(), elem_ty),
        }
    }

    fn infer_lit_pat(&mut self, expr: ExprId, expected: Ty<'db>) -> Ty<'db> {
        // Like slice patterns, byte string patterns can denote both `&[u8; N]` and `&[u8]`.
        if let Expr::Literal(Literal::ByteString(_)) = self.body[expr]
            && let TyKind::Ref(_, inner, _) = expected.kind()
        {
            let inner = self.table.try_structurally_resolve_type(inner);
            if matches!(inner.kind(), TyKind::Slice(_)) {
                let elem_ty = self.types.u8;
                let slice_ty = Ty::new_slice(self.interner(), elem_ty);
                let ty =
                    Ty::new_ref(self.interner(), self.types.re_static, slice_ty, Mutability::Not);
                self.write_expr_ty(expr, ty);
                return ty;
            }
        }

        self.infer_expr(expr, &Expectation::has_type(expected), ExprIsRead::Yes)
    }

    fn is_non_ref_pat(&mut self, body: &hir_def::expr_store::Body, pat: PatId) -> bool {
        match &body[pat] {
            Pat::Tuple { .. }
            | Pat::TupleStruct { .. }
            | Pat::Record { .. }
            | Pat::Range { .. }
            | Pat::Slice { .. } => true,
            Pat::Or(pats) => pats.iter().all(|p| self.is_non_ref_pat(body, *p)),
            Pat::Path(path) => {
                // A const is a reference pattern, but other value ns things aren't (see #16131).
                let resolved = self.resolve_value_path_inner(path, pat.into(), true);
                resolved.is_some_and(|it| !matches!(it.0, hir_def::resolver::ValueNs::ConstId(_)))
            }
            Pat::ConstBlock(..) => false,
            Pat::Lit(expr) => !matches!(
                body[*expr],
                Expr::Literal(Literal::String(..) | Literal::CString(..) | Literal::ByteString(..))
            ),
            Pat::Wild
            | Pat::Bind { .. }
            | Pat::Ref { .. }
            | Pat::Box { .. }
            | Pat::Missing
            | Pat::Expr(_) => false,
        }
    }

    fn try_resolve_slice_ty_to_array_ty(
        &mut self,
        before: &[PatId],
        suffix: &[PatId],
        slice: Option<PatId>,
    ) -> Option<Ty<'db>> {
        if slice.is_some() {
            return None;
        }

        let len = before.len() + suffix.len();
        let size = consteval::usize_const(self.db, Some(len as u128), self.owner.krate(self.db));

        let elem_ty = self.table.next_ty_var();
        let array_ty = Ty::new_array_with_const_len(self.interner(), elem_ty, size);
        Some(array_ty)
    }

    /// Used to determine whether we can infer the expected type in the slice pattern to be of type array.
    /// This is only possible if we're in an irrefutable pattern. If we were to allow this in refutable
    /// patterns we wouldn't e.g. report ambiguity in the following situation:
    ///
    /// ```ignore(rust)
    ///    struct Zeroes;
    ///    const ARR: [usize; 2] = [0; 2];
    ///    const ARR2: [usize; 2] = [2; 2];
    ///
    ///    impl Into<&'static [usize; 2]> for Zeroes {
    ///        fn into(self) -> &'static [usize; 2] {
    ///            &ARR
    ///        }
    ///    }
    ///
    ///    impl Into<&'static [usize]> for Zeroes {
    ///        fn into(self) -> &'static [usize] {
    ///            &ARR2
    ///        }
    ///    }
    ///
    ///    fn main() {
    ///        let &[a, b]: &[usize] = Zeroes.into() else {
    ///           ..
    ///        };
    ///    }
    /// ```
    ///
    /// If we're in an irrefutable pattern we prefer the array impl candidate given that
    /// the slice impl candidate would be rejected anyway (if no ambiguity existed).
    fn pat_is_irrefutable(&self, decl_ctxt: Option<DeclContext>) -> bool {
        matches!(decl_ctxt, Some(DeclContext { origin: DeclOrigin::LocalDecl { has_else: false } }))
    }
}

pub(super) fn contains_explicit_ref_binding(body: &Body, pat_id: PatId) -> bool {
    let mut res = false;
    body.walk_pats(pat_id, &mut |pat| {
        res |= matches!(body[pat], Pat::Bind { id, .. } if body[id].mode == BindingAnnotation::Ref);
    });
    res
}
