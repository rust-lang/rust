use crate::{errors, FnCtxt, RawTy};
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{
    pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
    MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::{HirId, Pat, PatKind};
use rustc_infer::infer;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::ty::{self, Adt, BindingMode, Ty, TypeVisitableExt};
use rustc_session::lint::builtin::NON_EXHAUSTIVE_OMITTED_PATTERNS;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::source_map::{Span, Spanned};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{BytePos, DUMMY_SP};
use rustc_target::abi::FieldIdx;
use rustc_trait_selection::traits::{ObligationCause, Pattern};
use ty::VariantDef;

use std::cmp;
use std::collections::hash_map::Entry::{Occupied, Vacant};

use super::report_unexpected_variant_res;

const CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ: &str = "\
This error indicates that a pointer to a trait type cannot be implicitly dereferenced by a \
pattern. Every trait defines a type, but because the size of trait implementors isn't fixed, \
this type has no compile-time size. Therefore, all accesses to trait types must be through \
pointers. If you encounter this error you should try to avoid dereferencing the pointer.

You can read more about trait objects in the Trait Objects section of the Reference: \
https://doc.rust-lang.org/reference/types.html#trait-objects";

fn is_number(text: &str) -> bool {
    text.chars().all(|c: char| c.is_digit(10))
}

/// Information about the expected type at the top level of type checking a pattern.
///
/// **NOTE:** This is only for use by diagnostics. Do NOT use for type checking logic!
#[derive(Copy, Clone)]
struct TopInfo<'tcx> {
    /// The `expected` type at the top level of type checking a pattern.
    expected: Ty<'tcx>,
    /// Was the origin of the `span` from a scrutinee expression?
    ///
    /// Otherwise there is no scrutinee and it could be e.g. from the type of a formal parameter.
    origin_expr: Option<&'tcx hir::Expr<'tcx>>,
    /// The span giving rise to the `expected` type, if one could be provided.
    ///
    /// If `origin_expr` is `true`, then this is the span of the scrutinee as in:
    ///
    /// - `match scrutinee { ... }`
    /// - `let _ = scrutinee;`
    ///
    /// This is used to point to add context in type errors.
    /// In the following example, `span` corresponds to the `a + b` expression:
    ///
    /// ```text
    /// error[E0308]: mismatched types
    ///  --> src/main.rs:L:C
    ///   |
    /// L |    let temp: usize = match a + b {
    ///   |                            ----- this expression has type `usize`
    /// L |         Ok(num) => num,
    ///   |         ^^^^^^^ expected `usize`, found enum `std::result::Result`
    ///   |
    ///   = note: expected type `usize`
    ///              found type `std::result::Result<_, _>`
    /// ```
    span: Option<Span>,
}

impl<'tcx> FnCtxt<'_, 'tcx> {
    fn pattern_cause(&self, ti: TopInfo<'tcx>, cause_span: Span) -> ObligationCause<'tcx> {
        let code =
            Pattern { span: ti.span, root_ty: ti.expected, origin_expr: ti.origin_expr.is_some() };
        self.cause(cause_span, code)
    }

    fn demand_eqtype_pat_diag(
        &self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        let mut diag =
            self.demand_eqtype_with_origin(&self.pattern_cause(ti, cause_span), expected, actual)?;
        if let Some(expr) = ti.origin_expr {
            self.suggest_fn_call(&mut diag, expr, expected, |output| {
                self.can_eq(self.param_env, output, actual)
            });
        }
        Some(diag)
    }

    fn demand_eqtype_pat(
        &self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) {
        if let Some(mut err) = self.demand_eqtype_pat_diag(cause_span, expected, actual, ti) {
            err.emit();
        }
    }
}

const INITIAL_BM: BindingMode = BindingMode::BindByValue(hir::Mutability::Not);

/// Mode for adjusting the expected type and binding mode.
enum AdjustMode {
    /// Peel off all immediate reference types.
    Peel,
    /// Reset binding mode to the initial mode.
    Reset,
    /// Pass on the input binding mode and expected type.
    Pass,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Type check the given top level pattern against the `expected` type.
    ///
    /// If a `Some(span)` is provided and `origin_expr` holds,
    /// then the `span` represents the scrutinee's span.
    /// The scrutinee is found in e.g. `match scrutinee { ... }` and `let pat = scrutinee;`.
    ///
    /// Otherwise, `Some(span)` represents the span of a type expression
    /// which originated the `expected` type.
    pub fn check_pat_top(
        &self,
        pat: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        span: Option<Span>,
        origin_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) {
        let info = TopInfo { expected, origin_expr, span };
        self.check_pat(pat, expected, INITIAL_BM, info);
    }

    /// Type check the given `pat` against the `expected` type
    /// with the provided `def_bm` (default binding mode).
    ///
    /// Outside of this module, `check_pat_top` should always be used.
    /// Conversely, inside this module, `check_pat_top` should never be used.
    #[instrument(level = "debug", skip(self, ti))]
    fn check_pat(
        &self,
        pat: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) {
        let path_res = match &pat.kind {
            PatKind::Path(qpath) => {
                Some(self.resolve_ty_and_res_fully_qualified_call(qpath, pat.hir_id, pat.span))
            }
            _ => None,
        };
        let adjust_mode = self.calc_adjust_mode(pat, path_res.map(|(res, ..)| res));
        let (expected, def_bm) = self.calc_default_binding_mode(pat, expected, def_bm, adjust_mode);

        let ty = match pat.kind {
            PatKind::Wild => expected,
            PatKind::Lit(lt) => self.check_pat_lit(pat.span, lt, expected, ti),
            PatKind::Range(lhs, rhs, _) => self.check_pat_range(pat.span, lhs, rhs, expected, ti),
            PatKind::Binding(ba, var_id, _, sub) => {
                self.check_pat_ident(pat, ba, var_id, sub, expected, def_bm, ti)
            }
            PatKind::TupleStruct(ref qpath, subpats, ddpos) => {
                self.check_pat_tuple_struct(pat, qpath, subpats, ddpos, expected, def_bm, ti)
            }
            PatKind::Path(ref qpath) => {
                self.check_pat_path(pat, qpath, path_res.unwrap(), expected, ti)
            }
            PatKind::Struct(ref qpath, fields, has_rest_pat) => {
                self.check_pat_struct(pat, qpath, fields, has_rest_pat, expected, def_bm, ti)
            }
            PatKind::Or(pats) => {
                for pat in pats {
                    self.check_pat(pat, expected, def_bm, ti);
                }
                expected
            }
            PatKind::Tuple(elements, ddpos) => {
                self.check_pat_tuple(pat.span, elements, ddpos, expected, def_bm, ti)
            }
            PatKind::Box(inner) => self.check_pat_box(pat.span, inner, expected, def_bm, ti),
            PatKind::Ref(inner, mutbl) => {
                self.check_pat_ref(pat, inner, mutbl, expected, def_bm, ti)
            }
            PatKind::Slice(before, slice, after) => {
                self.check_pat_slice(pat.span, before, slice, after, expected, def_bm, ti)
            }
        };

        self.write_ty(pat.hir_id, ty);

        // (note_1): In most of the cases where (note_1) is referenced
        // (literals and constants being the exception), we relate types
        // using strict equality, even though subtyping would be sufficient.
        // There are a few reasons for this, some of which are fairly subtle
        // and which cost me (nmatsakis) an hour or two debugging to remember,
        // so I thought I'd write them down this time.
        //
        // 1. There is no loss of expressiveness here, though it does
        // cause some inconvenience. What we are saying is that the type
        // of `x` becomes *exactly* what is expected. This can cause unnecessary
        // errors in some cases, such as this one:
        //
        // ```
        // fn foo<'x>(x: &'x i32) {
        //    let a = 1;
        //    let mut z = x;
        //    z = &a;
        // }
        // ```
        //
        // The reason we might get an error is that `z` might be
        // assigned a type like `&'x i32`, and then we would have
        // a problem when we try to assign `&a` to `z`, because
        // the lifetime of `&a` (i.e., the enclosing block) is
        // shorter than `'x`.
        //
        // HOWEVER, this code works fine. The reason is that the
        // expected type here is whatever type the user wrote, not
        // the initializer's type. In this case the user wrote
        // nothing, so we are going to create a type variable `Z`.
        // Then we will assign the type of the initializer (`&'x i32`)
        // as a subtype of `Z`: `&'x i32 <: Z`. And hence we
        // will instantiate `Z` as a type `&'0 i32` where `'0` is
        // a fresh region variable, with the constraint that `'x : '0`.
        // So basically we're all set.
        //
        // Note that there are two tests to check that this remains true
        // (`regions-reassign-{match,let}-bound-pointer.rs`).
        //
        // 2. An outdated issue related to the old HIR borrowck. See the test
        // `regions-relate-bound-regions-on-closures-to-inference-variables.rs`,
    }

    /// Compute the new expected type and default binding mode from the old ones
    /// as well as the pattern form we are currently checking.
    fn calc_default_binding_mode(
        &self,
        pat: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        adjust_mode: AdjustMode,
    ) -> (Ty<'tcx>, BindingMode) {
        match adjust_mode {
            AdjustMode::Pass => (expected, def_bm),
            AdjustMode::Reset => (expected, INITIAL_BM),
            AdjustMode::Peel => self.peel_off_references(pat, expected, def_bm),
        }
    }

    /// How should the binding mode and expected type be adjusted?
    ///
    /// When the pattern is a path pattern, `opt_path_res` must be `Some(res)`.
    fn calc_adjust_mode(&self, pat: &'tcx Pat<'tcx>, opt_path_res: Option<Res>) -> AdjustMode {
        // When we perform destructuring assignment, we disable default match bindings, which are
        // unintuitive in this context.
        if !pat.default_binding_modes {
            return AdjustMode::Reset;
        }
        match &pat.kind {
            // Type checking these product-like types successfully always require
            // that the expected type be of those types and not reference types.
            PatKind::Struct(..)
            | PatKind::TupleStruct(..)
            | PatKind::Tuple(..)
            | PatKind::Box(_)
            | PatKind::Range(..)
            | PatKind::Slice(..) => AdjustMode::Peel,
            // String and byte-string literals result in types `&str` and `&[u8]` respectively.
            // All other literals result in non-reference types.
            // As a result, we allow `if let 0 = &&0 {}` but not `if let "foo" = &&"foo {}`.
            //
            // Call `resolve_vars_if_possible` here for inline const blocks.
            PatKind::Lit(lt) => match self.resolve_vars_if_possible(self.check_expr(lt)).kind() {
                ty::Ref(..) => AdjustMode::Pass,
                _ => AdjustMode::Peel,
            },
            PatKind::Path(_) => match opt_path_res.unwrap() {
                // These constants can be of a reference type, e.g. `const X: &u8 = &0;`.
                // Peeling the reference types too early will cause type checking failures.
                // Although it would be possible to *also* peel the types of the constants too.
                Res::Def(DefKind::Const | DefKind::AssocConst, _) => AdjustMode::Pass,
                // In the `ValueNS`, we have `SelfCtor(..) | Ctor(_, Const), _)` remaining which
                // could successfully compile. The former being `Self` requires a unit struct.
                // In either case, and unlike constants, the pattern itself cannot be
                // a reference type wherefore peeling doesn't give up any expressiveness.
                _ => AdjustMode::Peel,
            },
            // When encountering a `& mut? pat` pattern, reset to "by value".
            // This is so that `x` and `y` here are by value, as they appear to be:
            //
            // ```
            // match &(&22, &44) {
            //   (&x, &y) => ...
            // }
            // ```
            //
            // See issue #46688.
            PatKind::Ref(..) => AdjustMode::Reset,
            // A `_` pattern works with any expected type, so there's no need to do anything.
            PatKind::Wild
            // Bindings also work with whatever the expected type is,
            // and moreover if we peel references off, that will give us the wrong binding type.
            // Also, we can have a subpattern `binding @ pat`.
            // Each side of the `@` should be treated independently (like with OR-patterns).
            | PatKind::Binding(..)
            // An OR-pattern just propagates to each individual alternative.
            // This is maximally flexible, allowing e.g., `Some(mut x) | &Some(mut x)`.
            // In that example, `Some(mut x)` results in `Peel` whereas `&Some(mut x)` in `Reset`.
            | PatKind::Or(_) => AdjustMode::Pass,
        }
    }

    /// Peel off as many immediately nested `& mut?` from the expected type as possible
    /// and return the new expected type and binding default binding mode.
    /// The adjustments vector, if non-empty is stored in a table.
    fn peel_off_references(
        &self,
        pat: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        mut def_bm: BindingMode,
    ) -> (Ty<'tcx>, BindingMode) {
        let mut expected = self.resolve_vars_with_obligations(expected);

        // Peel off as many `&` or `&mut` from the scrutinee type as possible. For example,
        // for `match &&&mut Some(5)` the loop runs three times, aborting when it reaches
        // the `Some(5)` which is not of type Ref.
        //
        // For each ampersand peeled off, update the binding mode and push the original
        // type into the adjustments vector.
        //
        // See the examples in `ui/match-defbm*.rs`.
        let mut pat_adjustments = vec![];
        while let ty::Ref(_, inner_ty, inner_mutability) = *expected.kind() {
            debug!("inspecting {:?}", expected);

            debug!("current discriminant is Ref, inserting implicit deref");
            // Preserve the reference type. We'll need it later during THIR lowering.
            pat_adjustments.push(expected);

            expected = inner_ty;
            def_bm = ty::BindByReference(match def_bm {
                // If default binding mode is by value, make it `ref` or `ref mut`
                // (depending on whether we observe `&` or `&mut`).
                ty::BindByValue(_) |
                // When `ref mut`, stay a `ref mut` (on `&mut`) or downgrade to `ref` (on `&`).
                ty::BindByReference(hir::Mutability::Mut) => inner_mutability,
                // Once a `ref`, always a `ref`.
                // This is because a `& &mut` cannot mutate the underlying value.
                ty::BindByReference(m @ hir::Mutability::Not) => m,
            });
        }

        if !pat_adjustments.is_empty() {
            debug!("default binding mode is now {:?}", def_bm);
            self.inh
                .typeck_results
                .borrow_mut()
                .pat_adjustments_mut()
                .insert(pat.hir_id, pat_adjustments);
        }

        (expected, def_bm)
    }

    fn check_pat_lit(
        &self,
        span: Span,
        lt: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        // We've already computed the type above (when checking for a non-ref pat),
        // so avoid computing it again.
        let ty = self.node_ty(lt.hir_id);

        // Byte string patterns behave the same way as array patterns
        // They can denote both statically and dynamically-sized byte arrays.
        let mut pat_ty = ty;
        if let hir::ExprKind::Lit(Spanned { node: ast::LitKind::ByteStr(..), .. }) = lt.kind {
            let expected = self.structurally_resolve_type(span, expected);
            if let ty::Ref(_, inner_ty, _) = expected.kind()
                && matches!(inner_ty.kind(), ty::Slice(_))
            {
                let tcx = self.tcx;
                trace!(?lt.hir_id.local_id, "polymorphic byte string lit");
                self.typeck_results
                    .borrow_mut()
                    .treat_byte_string_as_slice
                    .insert(lt.hir_id.local_id);
                pat_ty = tcx.mk_imm_ref(tcx.lifetimes.re_static, tcx.mk_slice(tcx.types.u8));
            }
        }

        if self.tcx.features().string_deref_patterns && let hir::ExprKind::Lit(Spanned { node: ast::LitKind::Str(..), .. }) = lt.kind {
            let tcx = self.tcx;
            let expected = self.resolve_vars_if_possible(expected);
            pat_ty = match expected.kind() {
                ty::Adt(def, _) if Some(def.did()) == tcx.lang_items().string() => expected,
                ty::Str => tcx.mk_static_str(),
                _ => pat_ty,
            };
        }

        // Somewhat surprising: in this case, the subtyping relation goes the
        // opposite way as the other cases. Actually what we really want is not
        // a subtyping relation at all but rather that there exists a LUB
        // (so that they can be compared). However, in practice, constants are
        // always scalars or strings. For scalars subtyping is irrelevant,
        // and for strings `ty` is type is `&'static str`, so if we say that
        //
        //     &'static str <: expected
        //
        // then that's equivalent to there existing a LUB.
        let cause = self.pattern_cause(ti, span);
        if let Some(mut err) = self.demand_suptype_with_origin(&cause, expected, pat_ty) {
            err.emit_unless(
                ti.span
                    .filter(|&s| {
                        // In the case of `if`- and `while`-expressions we've already checked
                        // that `scrutinee: bool`. We know that the pattern is `true`,
                        // so an error here would be a duplicate and from the wrong POV.
                        s.is_desugaring(DesugaringKind::CondTemporary)
                    })
                    .is_some(),
            );
        }

        pat_ty
    }

    fn check_pat_range(
        &self,
        span: Span,
        lhs: Option<&'tcx hir::Expr<'tcx>>,
        rhs: Option<&'tcx hir::Expr<'tcx>>,
        expected: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let calc_side = |opt_expr: Option<&'tcx hir::Expr<'tcx>>| match opt_expr {
            None => None,
            Some(expr) => {
                let ty = self.check_expr(expr);
                // Check that the end-point is possibly of numeric or char type.
                // The early check here is not for correctness, but rather better
                // diagnostics (e.g. when `&str` is being matched, `expected` will
                // be peeled to `str` while ty here is still `&str`, if we don't
                // err early here, a rather confusing unification error will be
                // emitted instead).
                let fail =
                    !(ty.is_numeric() || ty.is_char() || ty.is_ty_var() || ty.references_error());
                Some((fail, ty, expr.span))
            }
        };
        let mut lhs = calc_side(lhs);
        let mut rhs = calc_side(rhs);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            // There exists a side that didn't meet our criteria that the end-point
            // be of a numeric or char type, as checked in `calc_side` above.
            let guar = self.emit_err_pat_range(span, lhs, rhs);
            return self.tcx.ty_error(guar);
        }

        // Unify each side with `expected`.
        // Subtyping doesn't matter here, as the value is some kind of scalar.
        let demand_eqtype = |x: &mut _, y| {
            if let Some((ref mut fail, x_ty, x_span)) = *x
                && let Some(mut err) = self.demand_eqtype_pat_diag(x_span, expected, x_ty, ti)
            {
                if let Some((_, y_ty, y_span)) = y {
                    self.endpoint_has_type(&mut err, y_span, y_ty);
                }
                err.emit();
                *fail = true;
            }
        };
        demand_eqtype(&mut lhs, rhs);
        demand_eqtype(&mut rhs, lhs);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            return self.tcx.ty_error_misc();
        }

        // Find the unified type and check if it's of numeric or char type again.
        // This check is needed if both sides are inference variables.
        // We require types to be resolved here so that we emit inference failure
        // rather than "_ is not a char or numeric".
        let ty = self.structurally_resolve_type(span, expected);
        if !(ty.is_numeric() || ty.is_char() || ty.references_error()) {
            if let Some((ref mut fail, _, _)) = lhs {
                *fail = true;
            }
            if let Some((ref mut fail, _, _)) = rhs {
                *fail = true;
            }
            let guar = self.emit_err_pat_range(span, lhs, rhs);
            return self.tcx.ty_error(guar);
        }
        ty
    }

    fn endpoint_has_type(&self, err: &mut Diagnostic, span: Span, ty: Ty<'_>) {
        if !ty.references_error() {
            err.span_label(span, format!("this is of type `{}`", ty));
        }
    }

    fn emit_err_pat_range(
        &self,
        span: Span,
        lhs: Option<(bool, Ty<'tcx>, Span)>,
        rhs: Option<(bool, Ty<'tcx>, Span)>,
    ) -> ErrorGuaranteed {
        let span = match (lhs, rhs) {
            (Some((true, ..)), Some((true, ..))) => span,
            (Some((true, _, sp)), _) => sp,
            (_, Some((true, _, sp))) => sp,
            _ => span_bug!(span, "emit_err_pat_range: no side failed or exists but still error?"),
        };
        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0029,
            "only `char` and numeric types are allowed in range patterns"
        );
        let msg = |ty| {
            let ty = self.resolve_vars_if_possible(ty);
            format!("this is of type `{}` but it should be `char` or numeric", ty)
        };
        let mut one_side_err = |first_span, first_ty, second: Option<(bool, Ty<'tcx>, Span)>| {
            err.span_label(first_span, msg(first_ty));
            if let Some((_, ty, sp)) = second {
                let ty = self.resolve_vars_if_possible(ty);
                self.endpoint_has_type(&mut err, sp, ty);
            }
        };
        match (lhs, rhs) {
            (Some((true, lhs_ty, lhs_sp)), Some((true, rhs_ty, rhs_sp))) => {
                err.span_label(lhs_sp, msg(lhs_ty));
                err.span_label(rhs_sp, msg(rhs_ty));
            }
            (Some((true, lhs_ty, lhs_sp)), rhs) => one_side_err(lhs_sp, lhs_ty, rhs),
            (lhs, Some((true, rhs_ty, rhs_sp))) => one_side_err(rhs_sp, rhs_ty, lhs),
            _ => span_bug!(span, "Impossible, verified above."),
        }
        if (lhs, rhs).references_error() {
            err.downgrade_to_delayed_bug();
        }
        if self.tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "In a match expression, only numbers and characters can be matched \
                    against a range. This is because the compiler checks that the range \
                    is non-empty at compile-time, and is unable to evaluate arbitrary \
                    comparison functions. If you want to capture values of an orderable \
                    type between two end-points, you can use a guard.",
            );
        }
        err.emit()
    }

    fn check_pat_ident(
        &self,
        pat: &'tcx Pat<'tcx>,
        ba: hir::BindingAnnotation,
        var_id: HirId,
        sub: Option<&'tcx Pat<'tcx>>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        // Determine the binding mode...
        let bm = match ba {
            hir::BindingAnnotation::NONE => def_bm,
            _ => BindingMode::convert(ba),
        };
        // ...and store it in a side table:
        self.inh.typeck_results.borrow_mut().pat_binding_modes_mut().insert(pat.hir_id, bm);

        debug!("check_pat_ident: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);

        let local_ty = self.local_ty(pat.span, pat.hir_id);
        let eq_ty = match bm {
            ty::BindByReference(mutbl) => {
                // If the binding is like `ref x | ref mut x`,
                // then `x` is assigned a value of type `&M T` where M is the
                // mutability and T is the expected type.
                //
                // `x` is assigned a value of type `&M T`, hence `&M T <: typeof(x)`
                // is required. However, we use equality, which is stronger.
                // See (note_1) for an explanation.
                self.new_ref_ty(pat.span, mutbl, expected)
            }
            // Otherwise, the type of x is the expected type `T`.
            ty::BindByValue(_) => {
                // As above, `T <: typeof(x)` is required, but we use equality, see (note_1).
                expected
            }
        };
        self.demand_eqtype_pat(pat.span, eq_ty, local_ty, ti);

        // If there are multiple arms, make sure they all agree on
        // what the type of the binding `x` ought to be.
        if var_id != pat.hir_id {
            self.check_binding_alt_eq_ty(ba, pat.span, var_id, local_ty, ti);
        }

        if let Some(p) = sub {
            self.check_pat(p, expected, def_bm, ti);
        }

        local_ty
    }

    fn check_binding_alt_eq_ty(
        &self,
        ba: hir::BindingAnnotation,
        span: Span,
        var_id: HirId,
        ty: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) {
        let var_ty = self.local_ty(span, var_id);
        if let Some(mut err) = self.demand_eqtype_pat_diag(span, var_ty, ty, ti) {
            let hir = self.tcx.hir();
            let var_ty = self.resolve_vars_with_obligations(var_ty);
            let msg = format!("first introduced with type `{var_ty}` here");
            err.span_label(hir.span(var_id), msg);
            let in_match = hir.parent_iter(var_id).any(|(_, n)| {
                matches!(
                    n,
                    hir::Node::Expr(hir::Expr {
                        kind: hir::ExprKind::Match(.., hir::MatchSource::Normal),
                        ..
                    })
                )
            });
            let pre = if in_match { "in the same arm, " } else { "" };
            err.note(format!("{}a binding must have the same type in all alternatives", pre));
            self.suggest_adding_missing_ref_or_removing_ref(
                &mut err,
                span,
                var_ty,
                self.resolve_vars_with_obligations(ty),
                ba,
            );
            err.emit();
        }
    }

    fn suggest_adding_missing_ref_or_removing_ref(
        &self,
        err: &mut Diagnostic,
        span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ba: hir::BindingAnnotation,
    ) {
        match (expected.kind(), actual.kind(), ba) {
            (ty::Ref(_, inner_ty, _), _, hir::BindingAnnotation::NONE)
                if self.can_eq(self.param_env, *inner_ty, actual) =>
            {
                err.span_suggestion_verbose(
                    span.shrink_to_lo(),
                    "consider adding `ref`",
                    "ref ",
                    Applicability::MaybeIncorrect,
                );
            }
            (_, ty::Ref(_, inner_ty, _), hir::BindingAnnotation::REF)
                if self.can_eq(self.param_env, expected, *inner_ty) =>
            {
                err.span_suggestion_verbose(
                    span.with_hi(span.lo() + BytePos(4)),
                    "consider removing `ref`",
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
            _ => (),
        }
    }

    // Precondition: pat is a Ref(_) pattern
    fn borrow_pat_suggestion(&self, err: &mut Diagnostic, pat: &Pat<'_>) {
        let tcx = self.tcx;
        if let PatKind::Ref(inner, mutbl) = pat.kind
        && let PatKind::Binding(_, _, binding, ..) = inner.kind {
            let binding_parent_id = tcx.hir().parent_id(pat.hir_id);
            let binding_parent = tcx.hir().get(binding_parent_id);
            debug!(?inner, ?pat, ?binding_parent);

            let mutability = match mutbl {
                ast::Mutability::Mut => "mut",
                ast::Mutability::Not => "",
            };

            let mut_var_suggestion = 'block: {
                if mutbl.is_not() {
                    break 'block None;
                }

                let ident_kind = match binding_parent {
                    hir::Node::Param(_) => "parameter",
                    hir::Node::Local(_) => "variable",
                    hir::Node::Arm(_) => "binding",

                    // Provide diagnostics only if the parent pattern is struct-like,
                    // i.e. where `mut binding` makes sense
                    hir::Node::Pat(Pat { kind, .. }) => match kind {
                        PatKind::Struct(..)
                        | PatKind::TupleStruct(..)
                        | PatKind::Or(..)
                        | PatKind::Tuple(..)
                        | PatKind::Slice(..) => "binding",

                        PatKind::Wild
                        | PatKind::Binding(..)
                        | PatKind::Path(..)
                        | PatKind::Box(..)
                        | PatKind::Ref(..)
                        | PatKind::Lit(..)
                        | PatKind::Range(..) => break 'block None,
                    },

                    // Don't provide suggestions in other cases
                    _ => break 'block None,
                };

                Some((
                    pat.span,
                    format!("to declare a mutable {ident_kind} use"),
                    format!("mut {binding}"),
                ))

            };

            match binding_parent {
                // Check that there is explicit type (ie this is not a closure param with inferred type)
                // so we don't suggest moving something to the type that does not exist
                hir::Node::Param(hir::Param { ty_span, .. }) if binding.span != *ty_span => {
                    err.multipart_suggestion_verbose(
                        format!("to take parameter `{binding}` by reference, move `&{mutability}` to the type"),
                        vec![
                            (pat.span.until(inner.span), "".to_owned()),
                            (ty_span.shrink_to_lo(), mutbl.ref_prefix_str().to_owned()),
                        ],
                        Applicability::MachineApplicable
                    );

                    if let Some((sp, msg, sugg)) = mut_var_suggestion {
                        err.span_note(sp, format!("{msg}: `{sugg}`"));
                    }
                }
                hir::Node::Pat(pt) if let PatKind::TupleStruct(_, pat_arr, _) = pt.kind => {
                    for i in pat_arr.iter() {
                        if let PatKind::Ref(the_ref, _) = i.kind
                        && let PatKind::Binding(mt, _, ident, _) = the_ref.kind {
                            let hir::BindingAnnotation(_, mtblty) = mt;
                            err.span_suggestion_verbose(
                                i.span,
                                format!("consider removing `&{mutability}` from the pattern"),
                                mtblty.prefix_str().to_string() + &ident.name.to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    if let Some((sp, msg, sugg)) = mut_var_suggestion {
                        err.span_note(sp, format!("{msg}: `{sugg}`"));
                    }
                }
                hir::Node::Param(_) | hir::Node::Arm(_) | hir::Node::Pat(_) => {
                    // rely on match ergonomics or it might be nested `&&pat`
                    err.span_suggestion_verbose(
                        pat.span.until(inner.span),
                        format!("consider removing `&{mutability}` from the pattern"),
                        "",
                        Applicability::MaybeIncorrect,
                    );

                    if let Some((sp, msg, sugg)) = mut_var_suggestion {
                        err.span_note(sp, format!("{msg}: `{sugg}`"));
                    }
                }
                _ if let Some((sp, msg, sugg)) = mut_var_suggestion => {
                    err.span_suggestion(sp, msg, sugg, Applicability::MachineApplicable);
                }
                _ => {} // don't provide suggestions in other cases #55175
            }
        }
    }

    pub fn check_dereferenceable(
        &self,
        span: Span,
        expected: Ty<'tcx>,
        inner: &Pat<'_>,
    ) -> Result<(), ErrorGuaranteed> {
        if let PatKind::Binding(..) = inner.kind
            && let Some(mt) = self.shallow_resolve(expected).builtin_deref(true)
            && let ty::Dynamic(..) = mt.ty.kind()
        {
            // This is "x = SomeTrait" being reduced from
            // "let &x = &SomeTrait" or "let box x = Box<SomeTrait>", an error.
            let type_str = self.ty_to_string(expected);
            let mut err = struct_span_err!(
                self.tcx.sess,
                span,
                E0033,
                "type `{}` cannot be dereferenced",
                type_str
            );
            err.span_label(span, format!("type `{type_str}` cannot be dereferenced"));
            if self.tcx.sess.teach(&err.get_code().unwrap()) {
                err.note(CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ);
            }
            return Err(err.emit());
        }
        Ok(())
    }

    fn check_pat_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &hir::QPath<'_>,
        fields: &'tcx [hir::PatField<'tcx>],
        has_rest_pat: bool,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        // Resolve the path and check the definition for errors.
        let (variant, pat_ty) = match self.check_struct_path(qpath, pat.hir_id) {
            Ok(data) => data,
            Err(guar) => {
                let err = self.tcx.ty_error(guar);
                for field in fields {
                    let ti = ti;
                    self.check_pat(field.pat, err, def_bm, ti);
                }
                return err;
            }
        };

        // Type-check the path.
        self.demand_eqtype_pat(pat.span, expected, pat_ty, ti);

        // Type-check subpatterns.
        if self.check_struct_pat_fields(pat_ty, &pat, variant, fields, has_rest_pat, def_bm, ti) {
            pat_ty
        } else {
            self.tcx.ty_error_misc()
        }
    }

    fn check_pat_path(
        &self,
        pat: &Pat<'tcx>,
        qpath: &hir::QPath<'_>,
        path_resolution: (Res, Option<RawTy<'tcx>>, &'tcx [hir::PathSegment<'tcx>]),
        expected: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        // We have already resolved the path.
        let (res, opt_ty, segments) = path_resolution;
        match res {
            Res::Err => {
                let e = tcx.sess.delay_span_bug(qpath.span(), "`Res::Err` but no error emitted");
                self.set_tainted_by_errors(e);
                return tcx.ty_error(e);
            }
            Res::Def(DefKind::AssocFn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::Variant, _) => {
                let expected = "unit struct, unit variant or constant";
                let e = report_unexpected_variant_res(tcx, res, qpath, pat.span, "E0533", expected);
                return tcx.ty_error(e);
            }
            Res::SelfCtor(..)
            | Res::Def(
                DefKind::Ctor(_, CtorKind::Const)
                | DefKind::Const
                | DefKind::AssocConst
                | DefKind::ConstParam,
                _,
            ) => {} // OK
            _ => bug!("unexpected pattern resolution: {:?}", res),
        }

        // Type-check the path.
        let (pat_ty, pat_res) =
            self.instantiate_value_path(segments, opt_ty, res, pat.span, pat.hir_id);
        if let Some(err) =
            self.demand_suptype_with_origin(&self.pattern_cause(ti, pat.span), expected, pat_ty)
        {
            self.emit_bad_pat_path(err, pat, res, pat_res, pat_ty, segments);
        }
        pat_ty
    }

    fn maybe_suggest_range_literal(
        &self,
        e: &mut Diagnostic,
        opt_def_id: Option<hir::def_id::DefId>,
        ident: Ident,
    ) -> bool {
        match opt_def_id {
            Some(def_id) => match self.tcx.hir().get_if_local(def_id) {
                Some(hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Const(_, body_id), ..
                })) => match self.tcx.hir().get(body_id.hir_id) {
                    hir::Node::Expr(expr) => {
                        if hir::is_range_literal(expr) {
                            let span = self.tcx.hir().span(body_id.hir_id);
                            if let Ok(snip) = self.tcx.sess.source_map().span_to_snippet(span) {
                                e.span_suggestion_verbose(
                                    ident.span,
                                    "you may want to move the range into the match block",
                                    snip,
                                    Applicability::MachineApplicable,
                                );
                                return true;
                            }
                        }
                    }
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        }
        false
    }

    fn emit_bad_pat_path(
        &self,
        mut e: DiagnosticBuilder<'_, ErrorGuaranteed>,
        pat: &hir::Pat<'tcx>,
        res: Res,
        pat_res: Res,
        pat_ty: Ty<'tcx>,
        segments: &'tcx [hir::PathSegment<'tcx>],
    ) {
        let pat_span = pat.span;
        if let Some(span) = self.tcx.hir().res_span(pat_res) {
            e.span_label(span, format!("{} defined here", res.descr()));
            if let [hir::PathSegment { ident, .. }] = &*segments {
                e.span_label(
                    pat_span,
                    format!(
                        "`{}` is interpreted as {} {}, not a new binding",
                        ident,
                        res.article(),
                        res.descr(),
                    ),
                );
                match self.tcx.hir().get_parent(pat.hir_id) {
                    hir::Node::PatField(..) => {
                        e.span_suggestion_verbose(
                            ident.span.shrink_to_hi(),
                            "bind the struct field to a different name instead",
                            format!(": other_{}", ident.as_str().to_lowercase()),
                            Applicability::HasPlaceholders,
                        );
                    }
                    _ => {
                        let (type_def_id, item_def_id) = match pat_ty.kind() {
                            Adt(def, _) => match res {
                                Res::Def(DefKind::Const, def_id) => (Some(def.did()), Some(def_id)),
                                _ => (None, None),
                            },
                            _ => (None, None),
                        };

                        let ranges = &[
                            self.tcx.lang_items().range_struct(),
                            self.tcx.lang_items().range_from_struct(),
                            self.tcx.lang_items().range_to_struct(),
                            self.tcx.lang_items().range_full_struct(),
                            self.tcx.lang_items().range_inclusive_struct(),
                            self.tcx.lang_items().range_to_inclusive_struct(),
                        ];
                        if type_def_id != None && ranges.contains(&type_def_id) {
                            if !self.maybe_suggest_range_literal(&mut e, item_def_id, *ident) {
                                let msg = "constants only support matching by type, \
                                    if you meant to match against a range of values, \
                                    consider using a range pattern like `min ..= max` in the match block";
                                e.note(msg);
                            }
                        } else {
                            let msg = "introduce a new binding instead";
                            let sugg = format!("other_{}", ident.as_str().to_lowercase());
                            e.span_suggestion(
                                ident.span,
                                msg,
                                sugg,
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                };
            }
        }
        e.emit();
    }

    fn check_pat_tuple_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &'tcx hir::QPath<'tcx>,
        subpats: &'tcx [Pat<'tcx>],
        ddpos: hir::DotDotPos,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let on_error = |e| {
            for pat in subpats {
                self.check_pat(pat, tcx.ty_error(e), def_bm, ti);
            }
        };
        let report_unexpected_res = |res: Res| {
            let expected = "tuple struct or tuple variant";
            let e = report_unexpected_variant_res(tcx, res, qpath, pat.span, "E0164", expected);
            on_error(e);
            e
        };

        // Resolve the path and check the definition for errors.
        let (res, opt_ty, segments) =
            self.resolve_ty_and_res_fully_qualified_call(qpath, pat.hir_id, pat.span);
        if res == Res::Err {
            let e = tcx.sess.delay_span_bug(pat.span, "`Res::Err` but no error emitted");
            self.set_tainted_by_errors(e);
            on_error(e);
            return tcx.ty_error(e);
        }

        // Type-check the path.
        let (pat_ty, res) =
            self.instantiate_value_path(segments, opt_ty, res, pat.span, pat.hir_id);
        if !pat_ty.is_fn() {
            let e = report_unexpected_res(res);
            return tcx.ty_error(e);
        }

        let variant = match res {
            Res::Err => {
                let e = tcx.sess.delay_span_bug(pat.span, "`Res::Err` but no error emitted");
                self.set_tainted_by_errors(e);
                on_error(e);
                return tcx.ty_error(e);
            }
            Res::Def(DefKind::AssocConst | DefKind::AssocFn, _) => {
                let e = report_unexpected_res(res);
                return tcx.ty_error(e);
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) => tcx.expect_variant_res(res),
            _ => bug!("unexpected pattern resolution: {:?}", res),
        };

        // Replace constructor type with constructed type for tuple struct patterns.
        let pat_ty = pat_ty.fn_sig(tcx).output();
        let pat_ty = pat_ty.no_bound_vars().expect("expected fn type");

        // Type-check the tuple struct pattern against the expected type.
        let diag = self.demand_eqtype_pat_diag(pat.span, expected, pat_ty, ti);
        let had_err = if let Some(mut err) = diag {
            err.emit();
            true
        } else {
            false
        };

        // Type-check subpatterns.
        if subpats.len() == variant.fields.len()
            || subpats.len() < variant.fields.len() && ddpos.as_opt_usize().is_some()
        {
            let ty::Adt(_, substs) = pat_ty.kind() else {
                bug!("unexpected pattern type {:?}", pat_ty);
            };
            for (i, subpat) in subpats.iter().enumerate_and_adjust(variant.fields.len(), ddpos) {
                let field = &variant.fields[FieldIdx::from_usize(i)];
                let field_ty = self.field_ty(subpat.span, field, substs);
                self.check_pat(subpat, field_ty, def_bm, ti);

                self.tcx.check_stability(
                    variant.fields[FieldIdx::from_usize(i)].did,
                    Some(pat.hir_id),
                    subpat.span,
                    None,
                );
            }
        } else {
            // Pattern has wrong number of fields.
            let e =
                self.e0023(pat.span, res, qpath, subpats, &variant.fields.raw, expected, had_err);
            on_error(e);
            return tcx.ty_error(e);
        }
        pat_ty
    }

    fn e0023(
        &self,
        pat_span: Span,
        res: Res,
        qpath: &hir::QPath<'_>,
        subpats: &'tcx [Pat<'tcx>],
        fields: &'tcx [ty::FieldDef],
        expected: Ty<'tcx>,
        had_err: bool,
    ) -> ErrorGuaranteed {
        let subpats_ending = pluralize!(subpats.len());
        let fields_ending = pluralize!(fields.len());

        let subpat_spans = if subpats.is_empty() {
            vec![pat_span]
        } else {
            subpats.iter().map(|p| p.span).collect()
        };
        let last_subpat_span = *subpat_spans.last().unwrap();
        let res_span = self.tcx.def_span(res.def_id());
        let def_ident_span = self.tcx.def_ident_span(res.def_id()).unwrap_or(res_span);
        let field_def_spans = if fields.is_empty() {
            vec![res_span]
        } else {
            fields.iter().map(|f| f.ident(self.tcx).span).collect()
        };
        let last_field_def_span = *field_def_spans.last().unwrap();

        let mut err = struct_span_err!(
            self.tcx.sess,
            MultiSpan::from_spans(subpat_spans),
            E0023,
            "this pattern has {} field{}, but the corresponding {} has {} field{}",
            subpats.len(),
            subpats_ending,
            res.descr(),
            fields.len(),
            fields_ending,
        );
        err.span_label(
            last_subpat_span,
            format!("expected {} field{}, found {}", fields.len(), fields_ending, subpats.len()),
        );
        if self.tcx.sess.source_map().is_multiline(qpath.span().between(last_subpat_span)) {
            err.span_label(qpath.span(), "");
        }
        if self.tcx.sess.source_map().is_multiline(def_ident_span.between(last_field_def_span)) {
            err.span_label(def_ident_span, format!("{} defined here", res.descr()));
        }
        for span in &field_def_spans[..field_def_spans.len() - 1] {
            err.span_label(*span, "");
        }
        err.span_label(
            last_field_def_span,
            format!("{} has {} field{}", res.descr(), fields.len(), fields_ending),
        );

        // Identify the case `Some(x, y)` where the expected type is e.g. `Option<(T, U)>`.
        // More generally, the expected type wants a tuple variant with one field of an
        // N-arity-tuple, e.g., `V_i((p_0, .., p_N))`. Meanwhile, the user supplied a pattern
        // with the subpatterns directly in the tuple variant pattern, e.g., `V_i(p_0, .., p_N)`.
        let missing_parentheses = match (&expected.kind(), fields, had_err) {
            // #67037: only do this if we could successfully type-check the expected type against
            // the tuple struct pattern. Otherwise the substs could get out of range on e.g.,
            // `let P() = U;` where `P != U` with `struct P<T>(T);`.
            (ty::Adt(_, substs), [field], false) => {
                let field_ty = self.field_ty(pat_span, field, substs);
                match field_ty.kind() {
                    ty::Tuple(fields) => fields.len() == subpats.len(),
                    _ => false,
                }
            }
            _ => false,
        };
        if missing_parentheses {
            let (left, right) = match subpats {
                // This is the zero case; we aim to get the "hi" part of the `QPath`'s
                // span as the "lo" and then the "hi" part of the pattern's span as the "hi".
                // This looks like:
                //
                // help: missing parentheses
                //   |
                // L |     let A(()) = A(());
                //   |          ^  ^
                [] => (qpath.span().shrink_to_hi(), pat_span),
                // Easy case. Just take the "lo" of the first sub-pattern and the "hi" of the
                // last sub-pattern. In the case of `A(x)` the first and last may coincide.
                // This looks like:
                //
                // help: missing parentheses
                //   |
                // L |     let A((x, y)) = A((1, 2));
                //   |           ^    ^
                [first, ..] => (first.span.shrink_to_lo(), subpats.last().unwrap().span),
            };
            err.multipart_suggestion(
                "missing parentheses",
                vec![(left, "(".to_string()), (right.shrink_to_hi(), ")".to_string())],
                Applicability::MachineApplicable,
            );
        } else if fields.len() > subpats.len() && pat_span != DUMMY_SP {
            let after_fields_span = pat_span.with_hi(pat_span.hi() - BytePos(1)).shrink_to_hi();
            let all_fields_span = match subpats {
                [] => after_fields_span,
                [field] => field.span,
                [first, .., last] => first.span.to(last.span),
            };

            // Check if all the fields in the pattern are wildcards.
            let all_wildcards = subpats.iter().all(|pat| matches!(pat.kind, PatKind::Wild));
            let first_tail_wildcard =
                subpats.iter().enumerate().fold(None, |acc, (pos, pat)| match (acc, &pat.kind) {
                    (None, PatKind::Wild) => Some(pos),
                    (Some(_), PatKind::Wild) => acc,
                    _ => None,
                });
            let tail_span = match first_tail_wildcard {
                None => after_fields_span,
                Some(0) => subpats[0].span.to(after_fields_span),
                Some(pos) => subpats[pos - 1].span.shrink_to_hi().to(after_fields_span),
            };

            // FIXME: heuristic-based suggestion to check current types for where to add `_`.
            let mut wildcard_sugg = vec!["_"; fields.len() - subpats.len()].join(", ");
            if !subpats.is_empty() {
                wildcard_sugg = String::from(", ") + &wildcard_sugg;
            }

            err.span_suggestion_verbose(
                after_fields_span,
                "use `_` to explicitly ignore each field",
                wildcard_sugg,
                Applicability::MaybeIncorrect,
            );

            // Only suggest `..` if more than one field is missing
            // or the pattern consists of all wildcards.
            if fields.len() - subpats.len() > 1 || all_wildcards {
                if subpats.is_empty() || all_wildcards {
                    err.span_suggestion_verbose(
                        all_fields_span,
                        "use `..` to ignore all fields",
                        "..",
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_suggestion_verbose(
                        tail_span,
                        "use `..` to ignore the rest of the fields",
                        ", ..",
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }

        err.emit()
    }

    fn check_pat_tuple(
        &self,
        span: Span,
        elements: &'tcx [Pat<'tcx>],
        ddpos: hir::DotDotPos,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let mut expected_len = elements.len();
        if ddpos.as_opt_usize().is_some() {
            // Require known type only when `..` is present.
            if let ty::Tuple(tys) = self.structurally_resolve_type(span, expected).kind() {
                expected_len = tys.len();
            }
        }
        let max_len = cmp::max(expected_len, elements.len());

        let element_tys_iter = (0..max_len).map(|_| {
            self.next_ty_var(
                // FIXME: `MiscVariable` for now -- obtaining the span and name information
                // from all tuple elements isn't trivial.
                TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span },
            )
        });
        let element_tys = tcx.mk_type_list_from_iter(element_tys_iter);
        let pat_ty = tcx.mk_tup(element_tys);
        if let Some(mut err) = self.demand_eqtype_pat_diag(span, expected, pat_ty, ti) {
            let reported = err.emit();
            // Walk subpatterns with an expected type of `err` in this case to silence
            // further errors being emitted when using the bindings. #50333
            let element_tys_iter = (0..max_len).map(|_| tcx.ty_error(reported));
            for (_, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.check_pat(elem, tcx.ty_error(reported), def_bm, ti);
            }
            tcx.mk_tup_from_iter(element_tys_iter)
        } else {
            for (i, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.check_pat(elem, element_tys[i], def_bm, ti);
            }
            pat_ty
        }
    }

    fn check_struct_pat_fields(
        &self,
        adt_ty: Ty<'tcx>,
        pat: &'tcx Pat<'tcx>,
        variant: &'tcx ty::VariantDef,
        fields: &'tcx [hir::PatField<'tcx>],
        has_rest_pat: bool,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let ty::Adt(adt, substs) = adt_ty.kind() else {
            span_bug!(pat.span, "struct pattern is not an ADT");
        };

        // Index the struct fields' types.
        let field_map = variant
            .fields
            .iter_enumerated()
            .map(|(i, field)| (field.ident(self.tcx).normalize_to_macros_2_0(), (i, field)))
            .collect::<FxHashMap<_, _>>();

        // Keep track of which fields have already appeared in the pattern.
        let mut used_fields = FxHashMap::default();
        let mut no_field_errors = true;

        let mut inexistent_fields = vec![];
        // Typecheck each field.
        for field in fields {
            let span = field.span;
            let ident = tcx.adjust_ident(field.ident, variant.def_id);
            let field_ty = match used_fields.entry(ident) {
                Occupied(occupied) => {
                    no_field_errors = false;
                    let guar = self.error_field_already_bound(span, field.ident, *occupied.get());
                    tcx.ty_error(guar)
                }
                Vacant(vacant) => {
                    vacant.insert(span);
                    field_map
                        .get(&ident)
                        .map(|(i, f)| {
                            self.write_field_index(field.hir_id, *i);
                            self.tcx.check_stability(f.did, Some(pat.hir_id), span, None);
                            self.field_ty(span, f, substs)
                        })
                        .unwrap_or_else(|| {
                            inexistent_fields.push(field);
                            no_field_errors = false;
                            tcx.ty_error_misc()
                        })
                }
            };

            self.check_pat(field.pat, field_ty, def_bm, ti);
        }

        let mut unmentioned_fields = variant
            .fields
            .iter()
            .map(|field| (field, field.ident(self.tcx).normalize_to_macros_2_0()))
            .filter(|(_, ident)| !used_fields.contains_key(ident))
            .collect::<Vec<_>>();

        let inexistent_fields_err = if !(inexistent_fields.is_empty() || variant.is_recovered())
            && !inexistent_fields.iter().any(|field| field.ident.name == kw::Underscore)
        {
            Some(self.error_inexistent_fields(
                adt.variant_descr(),
                &inexistent_fields,
                &mut unmentioned_fields,
                variant,
                substs,
            ))
        } else {
            None
        };

        // Require `..` if struct has non_exhaustive attribute.
        let non_exhaustive = variant.is_field_list_non_exhaustive() && !adt.did().is_local();
        if non_exhaustive && !has_rest_pat {
            self.error_foreign_non_exhaustive_spat(pat, adt.variant_descr(), fields.is_empty());
        }

        let mut unmentioned_err = None;
        // Report an error if an incorrect number of fields was specified.
        if adt.is_union() {
            if fields.len() != 1 {
                tcx.sess.emit_err(errors::UnionPatMultipleFields { span: pat.span });
            }
            if has_rest_pat {
                tcx.sess.emit_err(errors::UnionPatDotDot { span: pat.span });
            }
        } else if !unmentioned_fields.is_empty() {
            let accessible_unmentioned_fields: Vec<_> = unmentioned_fields
                .iter()
                .copied()
                .filter(|(field, _)| {
                    field.vis.is_accessible_from(tcx.parent_module(pat.hir_id), tcx)
                        && !matches!(
                            tcx.eval_stability(field.did, None, DUMMY_SP, None),
                            EvalResult::Deny { .. }
                        )
                        // We only want to report the error if it is hidden and not local
                        && !(tcx.is_doc_hidden(field.did) && !field.did.is_local())
                })
                .collect();

            if !has_rest_pat {
                if accessible_unmentioned_fields.is_empty() {
                    unmentioned_err = Some(self.error_no_accessible_fields(pat, fields));
                } else {
                    unmentioned_err = Some(self.error_unmentioned_fields(
                        pat,
                        &accessible_unmentioned_fields,
                        accessible_unmentioned_fields.len() != unmentioned_fields.len(),
                        fields,
                    ));
                }
            } else if non_exhaustive && !accessible_unmentioned_fields.is_empty() {
                self.lint_non_exhaustive_omitted_patterns(
                    pat,
                    &accessible_unmentioned_fields,
                    adt_ty,
                )
            }
        }
        match (inexistent_fields_err, unmentioned_err) {
            (Some(mut i), Some(mut u)) => {
                if let Some(mut e) = self.error_tuple_variant_as_struct_pat(pat, fields, variant) {
                    // We don't want to show the nonexistent fields error when this was
                    // `Foo { a, b }` when it should have been `Foo(a, b)`.
                    i.delay_as_bug();
                    u.delay_as_bug();
                    e.emit();
                } else {
                    i.emit();
                    u.emit();
                }
            }
            (None, Some(mut u)) => {
                if let Some(mut e) = self.error_tuple_variant_as_struct_pat(pat, fields, variant) {
                    u.delay_as_bug();
                    e.emit();
                } else {
                    u.emit();
                }
            }
            (Some(mut err), None) => {
                err.emit();
            }
            (None, None) if let Some(mut err) =
                    self.error_tuple_variant_index_shorthand(variant, pat, fields) =>
            {
                err.emit();
            }
            (None, None) => {}
        }
        no_field_errors
    }

    fn error_tuple_variant_index_shorthand(
        &self,
        variant: &VariantDef,
        pat: &'_ Pat<'_>,
        fields: &[hir::PatField<'_>],
    ) -> Option<DiagnosticBuilder<'_, ErrorGuaranteed>> {
        // if this is a tuple struct, then all field names will be numbers
        // so if any fields in a struct pattern use shorthand syntax, they will
        // be invalid identifiers (for example, Foo { 0, 1 }).
        if let (Some(CtorKind::Fn), PatKind::Struct(qpath, field_patterns, ..)) =
            (variant.ctor_kind(), &pat.kind)
        {
            let has_shorthand_field_name = field_patterns.iter().any(|field| field.is_shorthand);
            if has_shorthand_field_name {
                let path = rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| {
                    s.print_qpath(qpath, false)
                });
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    pat.span,
                    E0769,
                    "tuple variant `{path}` written as struct variant",
                );
                err.span_suggestion_verbose(
                    qpath.span().shrink_to_hi().to(pat.span.shrink_to_hi()),
                    "use the tuple variant pattern syntax instead",
                    format!("({})", self.get_suggested_tuple_struct_pattern(fields, variant)),
                    Applicability::MaybeIncorrect,
                );
                return Some(err);
            }
        }
        None
    }

    fn error_foreign_non_exhaustive_spat(&self, pat: &Pat<'_>, descr: &str, no_fields: bool) {
        let sess = self.tcx.sess;
        let sm = sess.source_map();
        let sp_brace = sm.end_point(pat.span);
        let sp_comma = sm.end_point(pat.span.with_hi(sp_brace.hi()));
        let sugg = if no_fields || sp_brace != sp_comma { ".. }" } else { ", .. }" };

        let mut err = struct_span_err!(
            sess,
            pat.span,
            E0638,
            "`..` required with {descr} marked as non-exhaustive",
        );
        err.span_suggestion_verbose(
            sp_comma,
            "add `..` at the end of the field list to ignore all other fields",
            sugg,
            Applicability::MachineApplicable,
        );
        err.emit();
    }

    fn error_field_already_bound(
        &self,
        span: Span,
        ident: Ident,
        other_field: Span,
    ) -> ErrorGuaranteed {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0025,
            "field `{}` bound multiple times in the pattern",
            ident
        )
        .span_label(span, format!("multiple uses of `{ident}` in pattern"))
        .span_label(other_field, format!("first use of `{ident}`"))
        .emit()
    }

    fn error_inexistent_fields(
        &self,
        kind_name: &str,
        inexistent_fields: &[&hir::PatField<'tcx>],
        unmentioned_fields: &mut Vec<(&'tcx ty::FieldDef, Ident)>,
        variant: &ty::VariantDef,
        substs: &'tcx ty::List<ty::subst::GenericArg<'tcx>>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let tcx = self.tcx;
        let (field_names, t, plural) = if inexistent_fields.len() == 1 {
            (format!("a field named `{}`", inexistent_fields[0].ident), "this", "")
        } else {
            (
                format!(
                    "fields named {}",
                    inexistent_fields
                        .iter()
                        .map(|field| format!("`{}`", field.ident))
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
                "these",
                "s",
            )
        };
        let spans = inexistent_fields.iter().map(|field| field.ident.span).collect::<Vec<_>>();
        let mut err = struct_span_err!(
            tcx.sess,
            spans,
            E0026,
            "{} `{}` does not have {}",
            kind_name,
            tcx.def_path_str(variant.def_id),
            field_names
        );
        if let Some(pat_field) = inexistent_fields.last() {
            err.span_label(
                pat_field.ident.span,
                format!(
                    "{} `{}` does not have {} field{}",
                    kind_name,
                    tcx.def_path_str(variant.def_id),
                    t,
                    plural
                ),
            );

            if unmentioned_fields.len() == 1 {
                let input =
                    unmentioned_fields.iter().map(|(_, field)| field.name).collect::<Vec<_>>();
                let suggested_name = find_best_match_for_name(&input, pat_field.ident.name, None);
                if let Some(suggested_name) = suggested_name {
                    err.span_suggestion(
                        pat_field.ident.span,
                        "a field with a similar name exists",
                        suggested_name,
                        Applicability::MaybeIncorrect,
                    );

                    // When we have a tuple struct used with struct we don't want to suggest using
                    // the (valid) struct syntax with numeric field names. Instead we want to
                    // suggest the expected syntax. We infer that this is the case by parsing the
                    // `Ident` into an unsized integer. The suggestion will be emitted elsewhere in
                    // `smart_resolve_context_dependent_help`.
                    if suggested_name.to_ident_string().parse::<usize>().is_err() {
                        // We don't want to throw `E0027` in case we have thrown `E0026` for them.
                        unmentioned_fields.retain(|&(_, x)| x.name != suggested_name);
                    }
                } else if inexistent_fields.len() == 1 {
                    match pat_field.pat.kind {
                        PatKind::Lit(expr)
                            if !self.can_coerce(
                                self.typeck_results.borrow().expr_ty(expr),
                                self.field_ty(
                                    unmentioned_fields[0].1.span,
                                    unmentioned_fields[0].0,
                                    substs,
                                ),
                            ) => {}
                        _ => {
                            let unmentioned_field = unmentioned_fields[0].1.name;
                            err.span_suggestion_short(
                                pat_field.ident.span,
                                format!(
                                    "`{}` has a field named `{}`",
                                    tcx.def_path_str(variant.def_id),
                                    unmentioned_field
                                ),
                                unmentioned_field.to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                }
            }
        }
        if tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "This error indicates that a struct pattern attempted to \
                 extract a nonexistent field from a struct. Struct fields \
                 are identified by the name used before the colon : so struct \
                 patterns should resemble the declaration of the struct type \
                 being matched.\n\n\
                 If you are using shorthand field patterns but want to refer \
                 to the struct field by a different name, you should rename \
                 it explicitly.",
            );
        }
        err
    }

    fn error_tuple_variant_as_struct_pat(
        &self,
        pat: &Pat<'_>,
        fields: &'tcx [hir::PatField<'tcx>],
        variant: &ty::VariantDef,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        if let (Some(CtorKind::Fn), PatKind::Struct(qpath, pattern_fields, ..)) =
            (variant.ctor_kind(), &pat.kind)
        {
            let is_tuple_struct_match = !pattern_fields.is_empty()
                && pattern_fields.iter().map(|field| field.ident.name.as_str()).all(is_number);
            if is_tuple_struct_match {
                return None;
            }

            let path = rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| {
                s.print_qpath(qpath, false)
            });
            let mut err = struct_span_err!(
                self.tcx.sess,
                pat.span,
                E0769,
                "tuple variant `{}` written as struct variant",
                path
            );
            let (sugg, appl) = if fields.len() == variant.fields.len() {
                (
                    self.get_suggested_tuple_struct_pattern(fields, variant),
                    Applicability::MachineApplicable,
                )
            } else {
                (
                    variant.fields.iter().map(|_| "_").collect::<Vec<&str>>().join(", "),
                    Applicability::MaybeIncorrect,
                )
            };
            err.span_suggestion_verbose(
                qpath.span().shrink_to_hi().to(pat.span.shrink_to_hi()),
                "use the tuple variant pattern syntax instead",
                format!("({})", sugg),
                appl,
            );
            return Some(err);
        }
        None
    }

    fn get_suggested_tuple_struct_pattern(
        &self,
        fields: &[hir::PatField<'_>],
        variant: &VariantDef,
    ) -> String {
        let variant_field_idents =
            variant.fields.iter().map(|f| f.ident(self.tcx)).collect::<Vec<Ident>>();
        fields
            .iter()
            .map(|field| {
                match self.tcx.sess.source_map().span_to_snippet(field.pat.span) {
                    Ok(f) => {
                        // Field names are numbers, but numbers
                        // are not valid identifiers
                        if variant_field_idents.contains(&field.ident) {
                            String::from("_")
                        } else {
                            f
                        }
                    }
                    Err(_) => rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| {
                        s.print_pat(field.pat)
                    }),
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Returns a diagnostic reporting a struct pattern which is missing an `..` due to
    /// inaccessible fields.
    ///
    /// ```text
    /// error: pattern requires `..` due to inaccessible fields
    ///   --> src/main.rs:10:9
    ///    |
    /// LL |     let foo::Foo {} = foo::Foo::default();
    ///    |         ^^^^^^^^^^^
    ///    |
    /// help: add a `..`
    ///    |
    /// LL |     let foo::Foo { .. } = foo::Foo::default();
    ///    |                  ^^^^^^
    /// ```
    fn error_no_accessible_fields(
        &self,
        pat: &Pat<'_>,
        fields: &'tcx [hir::PatField<'tcx>],
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = self
            .tcx
            .sess
            .struct_span_err(pat.span, "pattern requires `..` due to inaccessible fields");

        if let Some(field) = fields.last() {
            err.span_suggestion_verbose(
                field.span.shrink_to_hi(),
                "ignore the inaccessible and unused fields",
                ", ..",
                Applicability::MachineApplicable,
            );
        } else {
            let qpath_span = if let PatKind::Struct(qpath, ..) = &pat.kind {
                qpath.span()
            } else {
                bug!("`error_no_accessible_fields` called on non-struct pattern");
            };

            // Shrink the span to exclude the `foo:Foo` in `foo::Foo { }`.
            let span = pat.span.with_lo(qpath_span.shrink_to_hi().hi());
            err.span_suggestion_verbose(
                span,
                "ignore the inaccessible and unused fields",
                " { .. }",
                Applicability::MachineApplicable,
            );
        }
        err
    }

    /// Report that a pattern for a `#[non_exhaustive]` struct marked with `non_exhaustive_omitted_patterns`
    /// is not exhaustive enough.
    ///
    /// Nb: the partner lint for enums lives in `compiler/rustc_mir_build/src/thir/pattern/usefulness.rs`.
    fn lint_non_exhaustive_omitted_patterns(
        &self,
        pat: &Pat<'_>,
        unmentioned_fields: &[(&ty::FieldDef, Ident)],
        ty: Ty<'tcx>,
    ) {
        fn joined_uncovered_patterns(witnesses: &[&Ident]) -> String {
            const LIMIT: usize = 3;
            match witnesses {
                [] => bug!(),
                [witness] => format!("`{}`", witness),
                [head @ .., tail] if head.len() < LIMIT => {
                    let head: Vec<_> = head.iter().map(<_>::to_string).collect();
                    format!("`{}` and `{}`", head.join("`, `"), tail)
                }
                _ => {
                    let (head, tail) = witnesses.split_at(LIMIT);
                    let head: Vec<_> = head.iter().map(<_>::to_string).collect();
                    format!("`{}` and {} more", head.join("`, `"), tail.len())
                }
            }
        }
        let joined_patterns = joined_uncovered_patterns(
            &unmentioned_fields.iter().map(|(_, i)| i).collect::<Vec<_>>(),
        );

        self.tcx.struct_span_lint_hir(NON_EXHAUSTIVE_OMITTED_PATTERNS, pat.hir_id, pat.span, "some fields are not explicitly listed", |lint| {
        lint.span_label(pat.span, format!("field{} {} not listed", rustc_errors::pluralize!(unmentioned_fields.len()), joined_patterns));
        lint.help(
            "ensure that all fields are mentioned explicitly by adding the suggested fields",
        );
        lint.note(format!(
            "the pattern is of type `{}` and the `non_exhaustive_omitted_patterns` attribute was found",
            ty,
        ));

        lint
    });
    }

    /// Returns a diagnostic reporting a struct pattern which does not mention some fields.
    ///
    /// ```text
    /// error[E0027]: pattern does not mention field `bar`
    ///   --> src/main.rs:15:9
    ///    |
    /// LL |     let foo::Foo {} = foo::Foo::new();
    ///    |         ^^^^^^^^^^^ missing field `bar`
    /// ```
    fn error_unmentioned_fields(
        &self,
        pat: &Pat<'_>,
        unmentioned_fields: &[(&ty::FieldDef, Ident)],
        have_inaccessible_fields: bool,
        fields: &'tcx [hir::PatField<'tcx>],
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let inaccessible = if have_inaccessible_fields { " and inaccessible fields" } else { "" };
        let field_names = if unmentioned_fields.len() == 1 {
            format!("field `{}`{}", unmentioned_fields[0].1, inaccessible)
        } else {
            let fields = unmentioned_fields
                .iter()
                .map(|(_, name)| format!("`{}`", name))
                .collect::<Vec<String>>()
                .join(", ");
            format!("fields {}{}", fields, inaccessible)
        };
        let mut err = struct_span_err!(
            self.tcx.sess,
            pat.span,
            E0027,
            "pattern does not mention {}",
            field_names
        );
        err.span_label(pat.span, format!("missing {}", field_names));
        let len = unmentioned_fields.len();
        let (prefix, postfix, sp) = match fields {
            [] => match &pat.kind {
                PatKind::Struct(path, [], false) => {
                    (" { ", " }", path.span().shrink_to_hi().until(pat.span.shrink_to_hi()))
                }
                _ => return err,
            },
            [.., field] => {
                // Account for last field having a trailing comma or parse recovery at the tail of
                // the pattern to avoid invalid suggestion (#78511).
                let tail = field.span.shrink_to_hi().with_hi(pat.span.hi());
                match &pat.kind {
                    PatKind::Struct(..) => (", ", " }", tail),
                    _ => return err,
                }
            }
        };
        err.span_suggestion(
            sp,
            format!(
                "include the missing field{} in the pattern{}",
                pluralize!(len),
                if have_inaccessible_fields { " and ignore the inaccessible fields" } else { "" }
            ),
            format!(
                "{}{}{}{}",
                prefix,
                unmentioned_fields
                    .iter()
                    .map(|(_, name)| {
                        let field_name = name.to_string();
                        if is_number(&field_name) {
                            format!("{}: _", field_name)
                        } else {
                            field_name
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                if have_inaccessible_fields { ", .." } else { "" },
                postfix,
            ),
            Applicability::MachineApplicable,
        );
        err.span_suggestion(
            sp,
            format!(
                "if you don't care about {these} missing field{s}, you can explicitly ignore {them}",
                these = pluralize!("this", len),
                s = pluralize!(len),
                them = if len == 1 { "it" } else { "them" },
            ),
            format!("{}..{}", prefix, postfix),
            Applicability::MachineApplicable,
        );
        err
    }

    fn check_pat_box(
        &self,
        span: Span,
        inner: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let (box_ty, inner_ty) = match self.check_dereferenceable(span, expected, inner) {
            Ok(()) => {
                // Here, `demand::subtype` is good enough, but I don't
                // think any errors can be introduced by using `demand::eqtype`.
                let inner_ty = self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeInference,
                    span: inner.span,
                });
                let box_ty = tcx.mk_box(inner_ty);
                self.demand_eqtype_pat(span, expected, box_ty, ti);
                (box_ty, inner_ty)
            }
            Err(guar) => {
                let err = tcx.ty_error(guar);
                (err, err)
            }
        };
        self.check_pat(inner, inner_ty, def_bm, ti);
        box_ty
    }

    // Precondition: Pat is Ref(inner)
    fn check_pat_ref(
        &self,
        pat: &'tcx Pat<'tcx>,
        inner: &'tcx Pat<'tcx>,
        mutbl: hir::Mutability,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let expected = self.shallow_resolve(expected);
        let (ref_ty, inner_ty) = match self.check_dereferenceable(pat.span, expected, inner) {
            Ok(()) => {
                // `demand::subtype` would be good enough, but using `eqtype` turns
                // out to be equally general. See (note_1) for details.

                // Take region, inner-type from expected type if we can,
                // to avoid creating needless variables. This also helps with
                // the bad interactions of the given hack detailed in (note_1).
                debug!("check_pat_ref: expected={:?}", expected);
                match *expected.kind() {
                    ty::Ref(_, r_ty, r_mutbl) if r_mutbl == mutbl => (expected, r_ty),
                    _ => {
                        let inner_ty = self.next_ty_var(TypeVariableOrigin {
                            kind: TypeVariableOriginKind::TypeInference,
                            span: inner.span,
                        });
                        let ref_ty = self.new_ref_ty(pat.span, mutbl, inner_ty);
                        debug!("check_pat_ref: demanding {:?} = {:?}", expected, ref_ty);
                        let err = self.demand_eqtype_pat_diag(pat.span, expected, ref_ty, ti);

                        // Look for a case like `fn foo(&foo: u32)` and suggest
                        // `fn foo(foo: &u32)`
                        if let Some(mut err) = err {
                            self.borrow_pat_suggestion(&mut err, pat);
                            err.emit();
                        }
                        (ref_ty, inner_ty)
                    }
                }
            }
            Err(guar) => {
                let err = tcx.ty_error(guar);
                (err, err)
            }
        };
        self.check_pat(inner, inner_ty, def_bm, ti);
        ref_ty
    }

    /// Create a reference type with a fresh region variable.
    fn new_ref_ty(&self, span: Span, mutbl: hir::Mutability, ty: Ty<'tcx>) -> Ty<'tcx> {
        let region = self.next_region_var(infer::PatternRegion(span));
        let mt = ty::TypeAndMut { ty, mutbl };
        self.tcx.mk_ref(region, mt)
    }

    /// Type check a slice pattern.
    ///
    /// Syntactically, these look like `[pat_0, ..., pat_n]`.
    /// Semantically, we are type checking a pattern with structure:
    /// ```ignore (not-rust)
    /// [before_0, ..., before_n, (slice, after_0, ... after_n)?]
    /// ```
    /// The type of `slice`, if it is present, depends on the `expected` type.
    /// If `slice` is missing, then so is `after_i`.
    /// If `slice` is present, it can still represent 0 elements.
    fn check_pat_slice(
        &self,
        span: Span,
        before: &'tcx [Pat<'tcx>],
        slice: Option<&'tcx Pat<'tcx>>,
        after: &'tcx [Pat<'tcx>],
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let expected = self.structurally_resolve_type(span, expected);
        let (element_ty, opt_slice_ty, inferred) = match *expected.kind() {
            // An array, so we might have something like `let [a, b, c] = [0, 1, 2];`.
            ty::Array(element_ty, len) => {
                let min = before.len() as u64 + after.len() as u64;
                let (opt_slice_ty, expected) =
                    self.check_array_pat_len(span, element_ty, expected, slice, len, min);
                // `opt_slice_ty.is_none()` => `slice.is_none()`.
                // Note, though, that opt_slice_ty could be `Some(error_ty)`.
                assert!(opt_slice_ty.is_some() || slice.is_none());
                (element_ty, opt_slice_ty, expected)
            }
            ty::Slice(element_ty) => (element_ty, Some(expected), expected),
            // The expected type must be an array or slice, but was neither, so error.
            _ => {
                let guar = expected
                    .error_reported()
                    .err()
                    .unwrap_or_else(|| self.error_expected_array_or_slice(span, expected, ti));
                let err = self.tcx.ty_error(guar);
                (err, Some(err), err)
            }
        };

        // Type check all the patterns before `slice`.
        for elt in before {
            self.check_pat(elt, element_ty, def_bm, ti);
        }
        // Type check the `slice`, if present, against its expected type.
        if let Some(slice) = slice {
            self.check_pat(slice, opt_slice_ty.unwrap(), def_bm, ti);
        }
        // Type check the elements after `slice`, if present.
        for elt in after {
            self.check_pat(elt, element_ty, def_bm, ti);
        }
        inferred
    }

    /// Type check the length of an array pattern.
    ///
    /// Returns both the type of the variable length pattern (or `None`), and the potentially
    /// inferred array type. We only return `None` for the slice type if `slice.is_none()`.
    fn check_array_pat_len(
        &self,
        span: Span,
        element_ty: Ty<'tcx>,
        arr_ty: Ty<'tcx>,
        slice: Option<&'tcx Pat<'tcx>>,
        len: ty::Const<'tcx>,
        min_len: u64,
    ) -> (Option<Ty<'tcx>>, Ty<'tcx>) {
        let guar = if let Some(len) = len.try_eval_target_usize(self.tcx, self.param_env) {
            // Now we know the length...
            if slice.is_none() {
                // ...and since there is no variable-length pattern,
                // we require an exact match between the number of elements
                // in the array pattern and as provided by the matched type.
                if min_len == len {
                    return (None, arr_ty);
                }

                self.error_scrutinee_inconsistent_length(span, min_len, len)
            } else if let Some(pat_len) = len.checked_sub(min_len) {
                // The variable-length pattern was there,
                // so it has an array type with the remaining elements left as its size...
                return (Some(self.tcx.mk_array(element_ty, pat_len)), arr_ty);
            } else {
                // ...however, in this case, there were no remaining elements.
                // That is, the slice pattern requires more than the array type offers.
                self.error_scrutinee_with_rest_inconsistent_length(span, min_len, len)
            }
        } else if slice.is_none() {
            // We have a pattern with a fixed length,
            // which we can use to infer the length of the array.
            let updated_arr_ty = self.tcx.mk_array(element_ty, min_len);
            self.demand_eqtype(span, updated_arr_ty, arr_ty);
            return (None, updated_arr_ty);
        } else {
            // We have a variable-length pattern and don't know the array length.
            // This happens if we have e.g.,
            // `let [a, b, ..] = arr` where `arr: [T; N]` where `const N: usize`.
            self.error_scrutinee_unfixed_length(span)
        };

        // If we get here, we must have emitted an error.
        (Some(self.tcx.ty_error(guar)), arr_ty)
    }

    fn error_scrutinee_inconsistent_length(
        &self,
        span: Span,
        min_len: u64,
        size: u64,
    ) -> ErrorGuaranteed {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0527,
            "pattern requires {} element{} but array has {}",
            min_len,
            pluralize!(min_len),
            size,
        )
        .span_label(span, format!("expected {} element{}", size, pluralize!(size)))
        .emit()
    }

    fn error_scrutinee_with_rest_inconsistent_length(
        &self,
        span: Span,
        min_len: u64,
        size: u64,
    ) -> ErrorGuaranteed {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0528,
            "pattern requires at least {} element{} but array has {}",
            min_len,
            pluralize!(min_len),
            size,
        )
        .span_label(
            span,
            format!("pattern cannot match array of {} element{}", size, pluralize!(size),),
        )
        .emit()
    }

    fn error_scrutinee_unfixed_length(&self, span: Span) -> ErrorGuaranteed {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0730,
            "cannot pattern-match on an array without a fixed length",
        )
        .emit()
    }

    fn error_expected_array_or_slice(
        &self,
        span: Span,
        expected_ty: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> ErrorGuaranteed {
        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0529,
            "expected an array or slice, found `{expected_ty}`"
        );
        if let ty::Ref(_, ty, _) = expected_ty.kind()
            && let ty::Array(..) | ty::Slice(..) = ty.kind()
        {
            err.help("the semantics of slice patterns changed recently; see issue #62254");
        } else if self.autoderef(span, expected_ty)
            .any(|(ty, _)| matches!(ty.kind(), ty::Slice(..) | ty::Array(..)))
            && let Some(span) = ti.span
            && let Some(_) = ti.origin_expr
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span)
        {
            let ty = self.resolve_vars_if_possible(ti.expected);
            let is_slice_or_array_or_vector = self.is_slice_or_array_or_vector(ty);
            match is_slice_or_array_or_vector.1.kind() {
                ty::Adt(adt_def, _)
                    if self.tcx.is_diagnostic_item(sym::Option, adt_def.did())
                        || self.tcx.is_diagnostic_item(sym::Result, adt_def.did()) =>
                {
                    // Slicing won't work here, but `.as_deref()` might (issue #91328).
                    err.span_suggestion(
                        span,
                        "consider using `as_deref` here",
                        format!("{snippet}.as_deref()"),
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => ()
            }
            if is_slice_or_array_or_vector.0 {
                err.span_suggestion(
                    span,
                    "consider slicing here",
                    format!("{snippet}[..]"),
                    Applicability::MachineApplicable,
                );
            }
        }
        err.span_label(span, format!("pattern cannot match with input type `{expected_ty}`"));
        err.emit()
    }

    fn is_slice_or_array_or_vector(&self, ty: Ty<'tcx>) -> (bool, Ty<'tcx>) {
        match ty.kind() {
            ty::Adt(adt_def, _) if self.tcx.is_diagnostic_item(sym::Vec, adt_def.did()) => {
                (true, ty)
            }
            ty::Ref(_, ty, _) => self.is_slice_or_array_or_vector(*ty),
            ty::Slice(..) | ty::Array(..) => (true, ty),
            _ => (false, ty),
        }
    }
}
