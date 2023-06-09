//! Type inference for patterns.

use std::iter::repeat_with;

use chalk_ir::Mutability;
use hir_def::{
    body::Body,
    hir::{Binding, BindingAnnotation, BindingId, Expr, ExprId, ExprOrPatId, Literal, Pat, PatId},
    path::Path,
};
use hir_expand::name::Name;

use crate::{
    consteval::{try_const_usize, usize_const},
    infer::{BindingMode, Expectation, InferenceContext, TypeMismatch},
    lower::lower_to_chalk_mutability,
    primitive::UintTy,
    static_lifetime, Interner, Scalar, Substitution, Ty, TyBuilder, TyExt, TyKind,
};

/// Used to generalize patterns and assignee expressions.
pub(super) trait PatLike: Into<ExprOrPatId> + Copy {
    type BindingMode: Copy;

    fn infer(
        this: &mut InferenceContext<'_>,
        id: Self,
        expected_ty: &Ty,
        default_bm: Self::BindingMode,
    ) -> Ty;
}

impl PatLike for ExprId {
    type BindingMode = ();

    fn infer(
        this: &mut InferenceContext<'_>,
        id: Self,
        expected_ty: &Ty,
        (): Self::BindingMode,
    ) -> Ty {
        this.infer_assignee_expr(id, expected_ty)
    }
}

impl PatLike for PatId {
    type BindingMode = BindingMode;

    fn infer(
        this: &mut InferenceContext<'_>,
        id: Self,
        expected_ty: &Ty,
        default_bm: Self::BindingMode,
    ) -> Ty {
        this.infer_pat(id, expected_ty, default_bm)
    }
}

impl<'a> InferenceContext<'a> {
    /// Infers type for tuple struct pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_struct_pat_like<T: PatLike>(
        &mut self,
        path: Option<&Path>,
        expected: &Ty,
        default_bm: T::BindingMode,
        id: T,
        ellipsis: Option<usize>,
        subs: &[T],
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path, true);
        let var_data = def.map(|it| it.variant_data(self.db.upcast()));
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }
        self.unify(&ty, expected);

        let substs =
            ty.as_adt().map(|(_, s)| s.clone()).unwrap_or_else(|| Substitution::empty(Interner));

        let field_tys = def.map(|it| self.db.field_types(it)).unwrap_or_default();
        let (pre, post) = match ellipsis {
            Some(idx) => subs.split_at(idx),
            None => (subs, &[][..]),
        };
        let post_idx_offset = field_tys.iter().count().saturating_sub(post.len());

        let pre_iter = pre.iter().enumerate();
        let post_iter = (post_idx_offset..).zip(post.iter());
        for (i, &subpat) in pre_iter.chain(post_iter) {
            let expected_ty = var_data
                .as_ref()
                .and_then(|d| d.field(&Name::new_tuple_field(i)))
                .map_or(self.err_ty(), |field| {
                    field_tys[field].clone().substitute(Interner, &substs)
                });
            let expected_ty = self.normalize_associated_types_in(expected_ty);
            T::infer(self, subpat, &expected_ty, default_bm);
        }

        ty
    }

    /// Infers type for record pattern or its corresponding assignee expression.
    pub(super) fn infer_record_pat_like<T: PatLike>(
        &mut self,
        path: Option<&Path>,
        expected: &Ty,
        default_bm: T::BindingMode,
        id: T,
        subs: impl Iterator<Item = (Name, T)>,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path, false);
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }

        self.unify(&ty, expected);

        let substs =
            ty.as_adt().map(|(_, s)| s.clone()).unwrap_or_else(|| Substitution::empty(Interner));

        let field_tys = def.map(|it| self.db.field_types(it)).unwrap_or_default();
        let var_data = def.map(|it| it.variant_data(self.db.upcast()));

        for (name, inner) in subs {
            let expected_ty = var_data
                .as_ref()
                .and_then(|it| it.field(&name))
                .map_or(self.err_ty(), |f| field_tys[f].clone().substitute(Interner, &substs));
            let expected_ty = self.normalize_associated_types_in(expected_ty);

            T::infer(self, inner, &expected_ty, default_bm);
        }

        ty
    }

    /// Infers type for tuple pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_pat_like<T: PatLike>(
        &mut self,
        expected: &Ty,
        default_bm: T::BindingMode,
        ellipsis: Option<usize>,
        subs: &[T],
    ) -> Ty {
        let expected = self.resolve_ty_shallow(expected);
        let expectations = match expected.as_tuple() {
            Some(parameters) => &*parameters.as_slice(Interner),
            _ => &[],
        };

        let ((pre, post), n_uncovered_patterns) = match ellipsis {
            Some(idx) => (subs.split_at(idx), expectations.len().saturating_sub(subs.len())),
            None => ((&subs[..], &[][..]), 0),
        };
        let mut expectations_iter = expectations
            .iter()
            .cloned()
            .map(|a| a.assert_ty_ref(Interner).clone())
            .chain(repeat_with(|| self.table.new_type_var()));

        let mut inner_tys = Vec::with_capacity(n_uncovered_patterns + subs.len());

        inner_tys.extend(expectations_iter.by_ref().take(n_uncovered_patterns + subs.len()));

        // Process pre
        for (ty, pat) in inner_tys.iter_mut().zip(pre) {
            *ty = T::infer(self, *pat, ty, default_bm);
        }

        // Process post
        for (ty, pat) in inner_tys.iter_mut().skip(pre.len() + n_uncovered_patterns).zip(post) {
            *ty = T::infer(self, *pat, ty, default_bm);
        }

        TyKind::Tuple(inner_tys.len(), Substitution::from_iter(Interner, inner_tys))
            .intern(Interner)
    }

    pub(super) fn infer_top_pat(&mut self, pat: PatId, expected: &Ty) {
        self.infer_pat(pat, expected, BindingMode::default());
    }

    fn infer_pat(&mut self, pat: PatId, expected: &Ty, mut default_bm: BindingMode) -> Ty {
        let mut expected = self.resolve_ty_shallow(expected);

        if is_non_ref_pat(self.body, pat) {
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
        } else if let Pat::Ref { .. } = &self.body[pat] {
            cov_mark::hit!(match_ergonomics_ref);
            // When you encounter a `&pat` pattern, reset to Move.
            // This is so that `w` is by value: `let (_, &w) = &(1, &2);`
            default_bm = BindingMode::Move;
        }

        // Lose mutability.
        let default_bm = default_bm;
        let expected = expected;

        let ty = match &self.body[pat] {
            Pat::Tuple { args, ellipsis } => {
                self.infer_tuple_pat_like(&expected, default_bm, *ellipsis, args)
            }
            Pat::Or(pats) => {
                for pat in pats.iter() {
                    self.infer_pat(*pat, &expected, default_bm);
                }
                expected.clone()
            }
            &Pat::Ref { pat, mutability } => self.infer_ref_pat(
                pat,
                lower_to_chalk_mutability(mutability),
                &expected,
                default_bm,
            ),
            Pat::TupleStruct { path: p, args: subpats, ellipsis } => self
                .infer_tuple_struct_pat_like(
                    p.as_deref(),
                    &expected,
                    default_bm,
                    pat,
                    *ellipsis,
                    subpats,
                ),
            Pat::Record { path: p, args: fields, ellipsis: _ } => {
                let subs = fields.iter().map(|f| (f.name.clone(), f.pat));
                self.infer_record_pat_like(p.as_deref(), &expected, default_bm, pat, subs)
            }
            Pat::Path(path) => {
                // FIXME update resolver for the surrounding expression
                self.infer_path(path, pat.into()).unwrap_or_else(|| self.err_ty())
            }
            Pat::Bind { id, subpat } => {
                return self.infer_bind_pat(pat, *id, default_bm, *subpat, &expected);
            }
            Pat::Slice { prefix, slice, suffix } => {
                self.infer_slice_pat(&expected, prefix, slice, suffix, default_bm)
            }
            Pat::Wild => expected.clone(),
            Pat::Range { .. } => {
                // FIXME: do some checks here.
                expected.clone()
            }
            &Pat::Lit(expr) => {
                // Don't emit type mismatches again, the expression lowering already did that.
                let ty = self.infer_lit_pat(expr, &expected);
                self.write_pat_ty(pat, ty.clone());
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

                    let inner_ty = self.infer_pat(*inner, &inner_ty, default_bm);
                    let mut b = TyBuilder::adt(self.db, box_adt).push(inner_ty);

                    if let Some(alloc_ty) = alloc_ty {
                        b = b.push(alloc_ty);
                    }
                    b.fill_with_defaults(self.db, || self.table.new_type_var()).build()
                }
                None => self.err_ty(),
            },
            Pat::ConstBlock(expr) => {
                self.infer_expr(*expr, &Expectation::has_type(expected.clone()))
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
            .and_then(|x| x.first())
            .unwrap_or(&self.result.type_of_pat[pat])
            .clone()
    }

    fn infer_ref_pat(
        &mut self,
        inner_pat: PatId,
        mutability: Mutability,
        expected: &Ty,
        default_bm: BindingMode,
    ) -> Ty {
        let expectation = match expected.as_reference() {
            Some((inner_ty, _lifetime, _exp_mut)) => inner_ty.clone(),
            None => {
                let inner_ty = self.table.new_type_var();
                let ref_ty =
                    TyKind::Ref(mutability, static_lifetime(), inner_ty.clone()).intern(Interner);
                // Unification failure will be reported by the caller.
                self.unify(&ref_ty, expected);
                inner_ty
            }
        };
        let subty = self.infer_pat(inner_pat, &expectation, default_bm);
        TyKind::Ref(mutability, static_lifetime(), subty).intern(Interner)
    }

    fn infer_bind_pat(
        &mut self,
        pat: PatId,
        binding: BindingId,
        default_bm: BindingMode,
        subpat: Option<PatId>,
        expected: &Ty,
    ) -> Ty {
        let Binding { mode, .. } = self.body.bindings[binding];
        let mode = if mode == BindingAnnotation::Unannotated {
            default_bm
        } else {
            BindingMode::convert(mode)
        };
        self.result.binding_modes.insert(binding, mode);

        let inner_ty = match subpat {
            Some(subpat) => self.infer_pat(subpat, &expected, default_bm),
            None => expected.clone(),
        };
        let inner_ty = self.insert_type_vars_shallow(inner_ty);

        let bound_ty = match mode {
            BindingMode::Ref(mutability) => {
                TyKind::Ref(mutability, static_lifetime(), inner_ty.clone()).intern(Interner)
            }
            BindingMode::Move => inner_ty.clone(),
        };
        self.write_pat_ty(pat, inner_ty.clone());
        self.write_binding_ty(binding, bound_ty);
        return inner_ty;
    }

    fn infer_slice_pat(
        &mut self,
        expected: &Ty,
        prefix: &[PatId],
        slice: &Option<PatId>,
        suffix: &[PatId],
        default_bm: BindingMode,
    ) -> Ty {
        let elem_ty = match expected.kind(Interner) {
            TyKind::Array(st, _) | TyKind::Slice(st) => st.clone(),
            _ => self.err_ty(),
        };

        for &pat_id in prefix.iter().chain(suffix.iter()) {
            self.infer_pat(pat_id, &elem_ty, default_bm);
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
            self.infer_pat(slice_pat_id, &rest_pat_ty, default_bm);
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

        self.infer_expr(expr, &Expectation::has_type(expected.clone()))
    }
}

fn is_non_ref_pat(body: &hir_def::body::Body, pat: PatId) -> bool {
    match &body[pat] {
        Pat::Tuple { .. }
        | Pat::TupleStruct { .. }
        | Pat::Record { .. }
        | Pat::Range { .. }
        | Pat::Slice { .. } => true,
        Pat::Or(pats) => pats.iter().all(|p| is_non_ref_pat(body, *p)),
        // FIXME: ConstBlock/Path/Lit might actually evaluate to ref, but inference is unimplemented.
        Pat::Path(..) => true,
        Pat::ConstBlock(..) => true,
        Pat::Lit(expr) => !matches!(
            body[*expr],
            Expr::Literal(Literal::String(..) | Literal::CString(..) | Literal::ByteString(..))
        ),
        Pat::Wild | Pat::Bind { .. } | Pat::Ref { .. } | Pat::Box { .. } | Pat::Missing => false,
    }
}

pub(super) fn contains_explicit_ref_binding(body: &Body, pat_id: PatId) -> bool {
    let mut res = false;
    body.walk_pats(pat_id, &mut |pat| {
        res |= matches!(body[pat], Pat::Bind { id, .. } if body.bindings[id].mode == BindingAnnotation::Ref);
    });
    res
}
