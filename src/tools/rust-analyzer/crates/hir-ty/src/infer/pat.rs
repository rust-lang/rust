//! Type inference for patterns.

use std::iter::repeat_with;

use hir_def::{
    HasModule,
    expr_store::{Body, path::Path},
    hir::{Binding, BindingAnnotation, BindingId, Expr, ExprId, Literal, Pat, PatId},
};
use hir_expand::name::Name;
use stdx::TupleExt;

use crate::{
    DeclContext, DeclOrigin, InferenceDiagnostic, Interner, Mutability, Scalar, Substitution, Ty,
    TyBuilder, TyExt, TyKind,
    consteval::{self, try_const_usize, usize_const},
    infer::{
        BindingMode, Expectation, InferenceContext, TypeMismatch, coerce::CoerceNever,
        expr::ExprIsRead,
    },
    lower::lower_to_chalk_mutability,
    primitive::UintTy,
    static_lifetime,
};

impl InferenceContext<'_> {
    /// Infers type for tuple struct pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_struct_pat_like(
        &mut self,
        path: Option<&Path>,
        expected: &Ty,
        default_bm: BindingMode,
        id: PatId,
        ellipsis: Option<u32>,
        subs: &[PatId],
        decl: Option<DeclContext>,
    ) -> Ty {
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

        self.unify(&ty, expected);

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
                                let f = field_types[local_id].clone();
                                let expected_ty = match substs {
                                    Some(substs) => f.substitute(Interner, substs),
                                    None => f.substitute(Interner, &Substitution::empty(Interner)),
                                };
                                self.normalize_associated_types_in(expected_ty)
                            }
                            None => self.err_ty(),
                        }
                    };

                    self.infer_pat(subpat, &expected_ty, default_bm, decl);
                }
            }
            None => {
                let err_ty = self.err_ty();
                for &inner in subs {
                    self.infer_pat(inner, &err_ty, default_bm, decl);
                }
            }
        }

        ty
    }

    /// Infers type for record pattern or its corresponding assignee expression.
    pub(super) fn infer_record_pat_like(
        &mut self,
        path: Option<&Path>,
        expected: &Ty,
        default_bm: BindingMode,
        id: PatId,
        subs: impl ExactSizeIterator<Item = (Name, PatId)>,
        decl: Option<DeclContext>,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(id.into(), path, false);
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }

        self.unify(&ty, expected);

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
                                let f = field_types[local_id].clone();
                                let expected_ty = match substs {
                                    Some(substs) => f.substitute(Interner, substs),
                                    None => f.substitute(Interner, &Substitution::empty(Interner)),
                                };
                                self.normalize_associated_types_in(expected_ty)
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

                    self.infer_pat(inner, &expected_ty, default_bm, decl);
                }
            }
            None => {
                let err_ty = self.err_ty();
                for (_, inner) in subs {
                    self.infer_pat(inner, &err_ty, default_bm, decl);
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
        expected: &Ty,
        default_bm: BindingMode,
        ellipsis: Option<u32>,
        subs: &[PatId],
        decl: Option<DeclContext>,
    ) -> Ty {
        let expected = self.resolve_ty_shallow(expected);
        let expectations = match expected.as_tuple() {
            Some(parameters) => parameters.as_slice(Interner),
            _ => &[],
        };

        let ((pre, post), n_uncovered_patterns) = match ellipsis {
            Some(idx) => {
                (subs.split_at(idx as usize), expectations.len().saturating_sub(subs.len()))
            }
            None => ((subs, &[][..]), 0),
        };
        let mut expectations_iter = expectations
            .iter()
            .map(|a| a.assert_ty_ref(Interner).clone())
            .chain(repeat_with(|| self.table.new_type_var()));

        let mut inner_tys = Vec::with_capacity(n_uncovered_patterns + subs.len());

        inner_tys.extend(expectations_iter.by_ref().take(n_uncovered_patterns + subs.len()));

        // Process pre
        for (ty, pat) in inner_tys.iter_mut().zip(pre) {
            *ty = self.infer_pat(*pat, ty, default_bm, decl);
        }

        // Process post
        for (ty, pat) in inner_tys.iter_mut().skip(pre.len() + n_uncovered_patterns).zip(post) {
            *ty = self.infer_pat(*pat, ty, default_bm, decl);
        }

        TyKind::Tuple(inner_tys.len(), Substitution::from_iter(Interner, inner_tys))
            .intern(Interner)
    }

    /// The resolver needs to be updated to the surrounding expression when inside assignment
    /// (because there, `Pat::Path` can refer to a variable).
    pub(super) fn infer_top_pat(&mut self, pat: PatId, expected: &Ty, decl: Option<DeclContext>) {
        self.infer_pat(pat, expected, BindingMode::default(), decl);
    }

    fn infer_pat(
        &mut self,
        pat: PatId,
        expected: &Ty,
        mut default_bm: BindingMode,
        decl: Option<DeclContext>,
    ) -> Ty {
        let mut expected = self.resolve_ty_shallow(expected);

        if matches!(&self.body[pat], Pat::Ref { .. }) || self.inside_assignment {
            cov_mark::hit!(match_ergonomics_ref);
            // When you encounter a `&pat` pattern, reset to Move.
            // This is so that `w` is by value: `let (_, &w) = &(1, &2);`
            // Destructuring assignments also reset the binding mode and
            // don't do match ergonomics.
            default_bm = BindingMode::Move;
        } else if self.is_non_ref_pat(self.body, pat) {
            let mut pat_adjustments = Vec::new();
            while let Some((inner, _lifetime, mutability)) = expected.as_reference() {
                pat_adjustments.push(expected.clone());
                expected = self.resolve_ty_shallow(inner);
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
                self.infer_tuple_pat_like(&expected, default_bm, *ellipsis, args, decl)
            }
            Pat::Or(pats) => {
                for pat in pats.iter() {
                    self.infer_pat(*pat, &expected, default_bm, decl);
                }
                expected.clone()
            }
            &Pat::Ref { pat, mutability } => self.infer_ref_pat(
                pat,
                lower_to_chalk_mutability(mutability),
                &expected,
                default_bm,
                decl,
            ),
            Pat::TupleStruct { path: p, args: subpats, ellipsis } => self
                .infer_tuple_struct_pat_like(
                    p.as_deref(),
                    &expected,
                    default_bm,
                    pat,
                    *ellipsis,
                    subpats,
                    decl,
                ),
            Pat::Record { path: p, args: fields, ellipsis: _ } => {
                let subs = fields.iter().map(|f| (f.name.clone(), f.pat));
                self.infer_record_pat_like(p.as_deref(), &expected, default_bm, pat, subs, decl)
            }
            Pat::Path(path) => {
                let ty = self.infer_path(path, pat.into()).unwrap_or_else(|| self.err_ty());
                let ty_inserted_vars = self.insert_type_vars_shallow(ty.clone());
                match self.table.coerce(&expected, &ty_inserted_vars, CoerceNever::Yes) {
                    Ok((adjustments, coerced_ty)) => {
                        if !adjustments.is_empty() {
                            self.result
                                .pat_adjustments
                                .entry(pat)
                                .or_default()
                                .extend(adjustments.into_iter().map(|adjust| adjust.target));
                        }
                        self.write_pat_ty(pat, coerced_ty);
                        return self.pat_ty_after_adjustment(pat);
                    }
                    Err(_) => {
                        self.result.type_mismatches.insert(
                            pat.into(),
                            TypeMismatch {
                                expected: expected.clone(),
                                actual: ty_inserted_vars.clone(),
                            },
                        );
                        self.write_pat_ty(pat, ty);
                        // We return `expected` to prevent cascading errors. I guess an alternative is to
                        // not emit type mismatches for error types and emit an error type here.
                        return expected;
                    }
                }
            }
            Pat::Bind { id, subpat } => {
                return self.infer_bind_pat(pat, *id, default_bm, *subpat, &expected, decl);
            }
            Pat::Slice { prefix, slice, suffix } => {
                self.infer_slice_pat(&expected, prefix, slice, suffix, default_bm, decl)
            }
            Pat::Wild => expected.clone(),
            Pat::Range { .. } => {
                // FIXME: do some checks here.
                expected.clone()
            }
            &Pat::Lit(expr) => {
                // Don't emit type mismatches again, the expression lowering already did that.
                let ty = self.infer_lit_pat(expr, &expected);
                self.write_pat_ty(pat, ty);
                return self.pat_ty_after_adjustment(pat);
            }
            Pat::Box { inner } => match self.resolve_boxed_box() {
                Some(box_adt) => {
                    let (inner_ty, alloc_ty) = match expected.as_adt() {
                        Some((adt, subst)) if adt == box_adt => (
                            subst.at(Interner, 0).assert_ty_ref(Interner).clone(),
                            subst.as_slice(Interner).get(1).and_then(|a| a.ty(Interner).cloned()),
                        ),
                        _ => (self.result.standard_types.unknown.clone(), None),
                    };

                    let inner_ty = self.infer_pat(*inner, &inner_ty, default_bm, decl);
                    let mut b = TyBuilder::adt(self.db, box_adt).push(inner_ty);

                    if let Some(alloc_ty) = alloc_ty {
                        b = b.push(alloc_ty);
                    }
                    b.fill_with_defaults(self.db, || self.table.new_type_var()).build()
                }
                None => self.err_ty(),
            },
            Pat::ConstBlock(expr) => {
                let old_inside_assign = std::mem::replace(&mut self.inside_assignment, false);
                let result = self.infer_expr(
                    *expr,
                    &Expectation::has_type(expected.clone()),
                    ExprIsRead::Yes,
                );
                self.inside_assignment = old_inside_assign;
                result
            }
            Pat::Expr(expr) => {
                let old_inside_assign = std::mem::replace(&mut self.inside_assignment, false);
                // LHS of assignment doesn't constitute reads.
                let result = self.infer_expr_coerce(
                    *expr,
                    &Expectation::has_type(expected.clone()),
                    ExprIsRead::No,
                );
                // We are returning early to avoid the unifiability check below.
                let lhs_ty = self.insert_type_vars_shallow(result);
                let ty = match self.coerce(None, &expected, &lhs_ty, CoerceNever::Yes) {
                    Ok(ty) => ty,
                    Err(_) => {
                        self.result.type_mismatches.insert(
                            pat.into(),
                            TypeMismatch { expected: expected.clone(), actual: lhs_ty.clone() },
                        );
                        // `rhs_ty` is returned so no further type mismatches are
                        // reported because of this mismatch.
                        expected
                    }
                };
                self.write_pat_ty(pat, ty.clone());
                self.inside_assignment = old_inside_assign;
                return ty;
            }
            Pat::Missing => self.err_ty(),
        };
        // use a new type variable if we got error type here
        let ty = self.insert_type_vars_shallow(ty);
        // FIXME: This never check is odd, but required with out we do inference right now
        if !expected.is_never() && !self.unify(&ty, &expected) {
            self.result
                .type_mismatches
                .insert(pat.into(), TypeMismatch { expected, actual: ty.clone() });
        }
        self.write_pat_ty(pat, ty);
        self.pat_ty_after_adjustment(pat)
    }

    fn pat_ty_after_adjustment(&self, pat: PatId) -> Ty {
        self.result
            .pat_adjustments
            .get(&pat)
            .and_then(|it| it.first())
            .unwrap_or(&self.result.type_of_pat[pat])
            .clone()
    }

    fn infer_ref_pat(
        &mut self,
        inner_pat: PatId,
        mutability: Mutability,
        expected: &Ty,
        default_bm: BindingMode,
        decl: Option<DeclContext>,
    ) -> Ty {
        let (expectation_type, expectation_lt) = match expected.as_reference() {
            Some((inner_ty, lifetime, _exp_mut)) => (inner_ty.clone(), lifetime),
            None => {
                let inner_ty = self.table.new_type_var();
                let inner_lt = self.table.new_lifetime_var();
                let ref_ty =
                    TyKind::Ref(mutability, inner_lt.clone(), inner_ty.clone()).intern(Interner);
                // Unification failure will be reported by the caller.
                self.unify(&ref_ty, expected);
                (inner_ty, inner_lt)
            }
        };
        let subty = self.infer_pat(inner_pat, &expectation_type, default_bm, decl);
        TyKind::Ref(mutability, expectation_lt, subty).intern(Interner)
    }

    fn infer_bind_pat(
        &mut self,
        pat: PatId,
        binding: BindingId,
        default_bm: BindingMode,
        subpat: Option<PatId>,
        expected: &Ty,
        decl: Option<DeclContext>,
    ) -> Ty {
        let Binding { mode, .. } = self.body.bindings[binding];
        let mode = if mode == BindingAnnotation::Unannotated {
            default_bm
        } else {
            BindingMode::convert(mode)
        };
        self.result.binding_modes.insert(pat, mode);

        let inner_ty = match subpat {
            Some(subpat) => self.infer_pat(subpat, expected, default_bm, decl),
            None => expected.clone(),
        };
        let inner_ty = self.insert_type_vars_shallow(inner_ty);

        let bound_ty = match mode {
            BindingMode::Ref(mutability) => {
                let inner_lt = self.table.new_lifetime_var();
                TyKind::Ref(mutability, inner_lt, inner_ty.clone()).intern(Interner)
            }
            BindingMode::Move => inner_ty.clone(),
        };
        self.write_pat_ty(pat, inner_ty.clone());
        self.write_binding_ty(binding, bound_ty);
        inner_ty
    }

    fn infer_slice_pat(
        &mut self,
        expected: &Ty,
        prefix: &[PatId],
        slice: &Option<PatId>,
        suffix: &[PatId],
        default_bm: BindingMode,
        decl: Option<DeclContext>,
    ) -> Ty {
        let expected = self.resolve_ty_shallow(expected);

        // If `expected` is an infer ty, we try to equate it to an array if the given pattern
        // allows it. See issue #16609
        if self.pat_is_irrefutable(decl) && expected.is_ty_var() {
            if let Some(resolved_array_ty) =
                self.try_resolve_slice_ty_to_array_ty(prefix, suffix, slice)
            {
                self.unify(&expected, &resolved_array_ty);
            }
        }

        let expected = self.resolve_ty_shallow(&expected);
        let elem_ty = match expected.kind(Interner) {
            TyKind::Array(st, _) | TyKind::Slice(st) => st.clone(),
            _ => self.err_ty(),
        };

        for &pat_id in prefix.iter().chain(suffix.iter()) {
            self.infer_pat(pat_id, &elem_ty, default_bm, decl);
        }

        if let &Some(slice_pat_id) = slice {
            let rest_pat_ty = match expected.kind(Interner) {
                TyKind::Array(_, length) => {
                    let len = try_const_usize(self.db, length);
                    let len =
                        len.and_then(|len| len.checked_sub((prefix.len() + suffix.len()) as u128));
                    TyKind::Array(elem_ty.clone(), usize_const(self.db, len, self.resolver.krate()))
                }
                _ => TyKind::Slice(elem_ty.clone()),
            }
            .intern(Interner);
            self.infer_pat(slice_pat_id, &rest_pat_ty, default_bm, decl);
        }

        match expected.kind(Interner) {
            TyKind::Array(_, const_) => TyKind::Array(elem_ty, const_.clone()),
            _ => TyKind::Slice(elem_ty),
        }
        .intern(Interner)
    }

    fn infer_lit_pat(&mut self, expr: ExprId, expected: &Ty) -> Ty {
        // Like slice patterns, byte string patterns can denote both `&[u8; N]` and `&[u8]`.
        if let Expr::Literal(Literal::ByteString(_)) = self.body[expr] {
            if let Some((inner, ..)) = expected.as_reference() {
                let inner = self.resolve_ty_shallow(inner);
                if matches!(inner.kind(Interner), TyKind::Slice(_)) {
                    let elem_ty = TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(Interner);
                    let slice_ty = TyKind::Slice(elem_ty).intern(Interner);
                    let ty =
                        TyKind::Ref(Mutability::Not, static_lifetime(), slice_ty).intern(Interner);
                    self.write_expr_ty(expr, ty.clone());
                    return ty;
                }
            }
        }

        self.infer_expr(expr, &Expectation::has_type(expected.clone()), ExprIsRead::Yes)
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
        slice: &Option<PatId>,
    ) -> Option<Ty> {
        if !slice.is_none() {
            return None;
        }

        let len = before.len() + suffix.len();
        let size = consteval::usize_const(self.db, Some(len as u128), self.owner.krate(self.db));

        let elem_ty = self.table.new_type_var();
        let array_ty = TyKind::Array(elem_ty, size).intern(Interner);
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
        res |= matches!(body[pat], Pat::Bind { id, .. } if body.bindings[id].mode == BindingAnnotation::Ref);
    });
    res
}
