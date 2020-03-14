use crate::check::FnCtxt;
use rustc::ty::subst::GenericArg;
use rustc::ty::{self, BindingMode, Ty, TypeFoldable};
use rustc_ast::ast;
use rustc_ast::util::lev_distance::find_best_match_for_name;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::{HirId, Pat, PatKind};
use rustc_infer::infer;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_span::hygiene::DesugaringKind;
use rustc_span::source_map::{Span, Spanned};
use rustc_trait_selection::traits::{ObligationCause, Pattern};

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
    origin_expr: bool,
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
    /// This refers to the parent pattern. Used to provide extra diagnostic information on errors.
    /// ```text
    /// error[E0308]: mismatched types
    ///   --> $DIR/const-in-struct-pat.rs:8:17
    ///   |
    /// L | struct f;
    ///   | --------- unit struct defined here
    /// ...
    /// L |     let Thing { f } = t;
    ///   |                 ^
    ///   |                 |
    ///   |                 expected struct `std::string::String`, found struct `f`
    ///   |                 `f` is interpreted as a unit struct, not a new binding
    ///   |                 help: bind the struct field to a different name instead: `f: other_f`
    /// ```
    parent_pat: Option<&'tcx Pat<'tcx>>,
}

impl<'tcx> FnCtxt<'_, 'tcx> {
    fn pattern_cause(&self, ti: TopInfo<'tcx>, cause_span: Span) -> ObligationCause<'tcx> {
        let code = Pattern { span: ti.span, root_ty: ti.expected, origin_expr: ti.origin_expr };
        self.cause(cause_span, code)
    }

    fn demand_eqtype_pat_diag(
        &self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx>> {
        self.demand_eqtype_with_origin(&self.pattern_cause(ti, cause_span), expected, actual)
    }

    fn demand_eqtype_pat(
        &self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) {
        self.demand_eqtype_pat_diag(cause_span, expected, actual, ti).map(|mut err| err.emit());
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
        origin_expr: bool,
    ) {
        let info = TopInfo { expected, origin_expr, span, parent_pat: None };
        self.check_pat(pat, expected, INITIAL_BM, info);
    }

    /// Type check the given `pat` against the `expected` type
    /// with the provided `def_bm` (default binding mode).
    ///
    /// Outside of this module, `check_pat_top` should always be used.
    /// Conversely, inside this module, `check_pat_top` should never be used.
    fn check_pat(
        &self,
        pat: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) {
        debug!("check_pat(pat={:?},expected={:?},def_bm={:?})", pat, expected, def_bm);

        let path_res = match &pat.kind {
            PatKind::Path(qpath) => Some(self.resolve_ty_and_res_ufcs(qpath, pat.hir_id, pat.span)),
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
                self.check_pat_path(pat, path_res.unwrap(), qpath, expected, ti)
            }
            PatKind::Struct(ref qpath, fields, etc) => {
                self.check_pat_struct(pat, qpath, fields, etc, expected, def_bm, ti)
            }
            PatKind::Or(pats) => {
                let parent_pat = Some(pat);
                for pat in pats {
                    self.check_pat(pat, expected, def_bm, TopInfo { parent_pat, ..ti });
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
        // fn foo<'x>(x: &'x int) {
        //    let a = 1;
        //    let mut z = x;
        //    z = &a;
        // }
        // ```
        //
        // The reason we might get an error is that `z` might be
        // assigned a type like `&'x int`, and then we would have
        // a problem when we try to assign `&a` to `z`, because
        // the lifetime of `&a` (i.e., the enclosing block) is
        // shorter than `'x`.
        //
        // HOWEVER, this code works fine. The reason is that the
        // expected type here is whatever type the user wrote, not
        // the initializer's type. In this case the user wrote
        // nothing, so we are going to create a type variable `Z`.
        // Then we will assign the type of the initializer (`&'x
        // int`) as a subtype of `Z`: `&'x int <: Z`. And hence we
        // will instantiate `Z` as a type `&'0 int` where `'0` is
        // a fresh region variable, with the constraint that `'x :
        // '0`.  So basically we're all set.
        //
        // Note that there are two tests to check that this remains true
        // (`regions-reassign-{match,let}-bound-pointer.rs`).
        //
        // 2. Things go horribly wrong if we use subtype. The reason for
        // THIS is a fairly subtle case involving bound regions. See the
        // `givens` field in `region_constraints`, as well as the test
        // `regions-relate-bound-regions-on-closures-to-inference-variables.rs`,
        // for details. Short version is that we must sometimes detect
        // relationships between specific region variables and regions
        // bound in a closure signature, and that detection gets thrown
        // off when we substitute fresh region variables here to enable
        // subtyping.
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
            PatKind::Lit(lt) => match self.check_expr(lt).kind {
                ty::Ref(..) => AdjustMode::Pass,
                _ => AdjustMode::Peel,
            },
            PatKind::Path(_) => match opt_path_res.unwrap() {
                // These constants can be of a reference type, e.g. `const X: &u8 = &0;`.
                // Peeling the reference types too early will cause type checking failures.
                // Although it would be possible to *also* peel the types of the constants too.
                Res::Def(DefKind::Const, _) | Res::Def(DefKind::AssocConst, _) => AdjustMode::Pass,
                // In the `ValueNS`, we have `SelfCtor(..) | Ctor(_, Const), _)` remaining which
                // could successfully compile. The former being `Self` requires a unit struct.
                // In either case, and unlike constants, the pattern itself cannot be
                // a reference type wherefore peeling doesn't give up any expressivity.
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
        let mut expected = self.resolve_vars_with_obligations(&expected);

        // Peel off as many `&` or `&mut` from the scrutinee type as possible. For example,
        // for `match &&&mut Some(5)` the loop runs three times, aborting when it reaches
        // the `Some(5)` which is not of type Ref.
        //
        // For each ampersand peeled off, update the binding mode and push the original
        // type into the adjustments vector.
        //
        // See the examples in `ui/match-defbm*.rs`.
        let mut pat_adjustments = vec![];
        while let ty::Ref(_, inner_ty, inner_mutability) = expected.kind {
            debug!("inspecting {:?}", expected);

            debug!("current discriminant is Ref, inserting implicit deref");
            // Preserve the reference type. We'll need it later during HAIR lowering.
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
            self.inh.tables.borrow_mut().pat_adjustments_mut().insert(pat.hir_id, pat_adjustments);
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
        if let hir::ExprKind::Lit(Spanned { node: ast::LitKind::ByteStr(_), .. }) = lt.kind {
            let expected = self.structurally_resolved_type(span, expected);
            if let ty::Ref(_, ty::TyS { kind: ty::Slice(_), .. }, _) = expected.kind {
                let tcx = self.tcx;
                pat_ty = tcx.mk_imm_ref(tcx.lifetimes.re_static, tcx.mk_slice(tcx.types.u8));
            }
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
            None => (None, None),
            Some(expr) => {
                let ty = self.check_expr(expr);
                // Check that the end-point is of numeric or char type.
                let fail = !(ty.is_numeric() || ty.is_char() || ty.references_error());
                (Some(ty), Some((fail, ty, expr.span)))
            }
        };
        let (lhs_ty, lhs) = calc_side(lhs);
        let (rhs_ty, rhs) = calc_side(rhs);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            // There exists a side that didn't meet our criteria that the end-point
            // be of a numeric or char type, as checked in `calc_side` above.
            self.emit_err_pat_range(span, lhs, rhs);
            return self.tcx.types.err;
        }

        // Now that we know the types can be unified we find the unified type
        // and use it to type the entire expression.
        let common_type = self.resolve_vars_if_possible(&lhs_ty.or(rhs_ty).unwrap_or(expected));

        // Subtyping doesn't matter here, as the value is some kind of scalar.
        let demand_eqtype = |x, y| {
            if let Some((_, x_ty, x_span)) = x {
                self.demand_eqtype_pat_diag(x_span, expected, x_ty, ti).map(|mut err| {
                    if let Some((_, y_ty, y_span)) = y {
                        self.endpoint_has_type(&mut err, y_span, y_ty);
                    }
                    err.emit();
                });
            }
        };
        demand_eqtype(lhs, rhs);
        demand_eqtype(rhs, lhs);

        common_type
    }

    fn endpoint_has_type(&self, err: &mut DiagnosticBuilder<'_>, span: Span, ty: Ty<'_>) {
        if !ty.references_error() {
            err.span_label(span, &format!("this is of type `{}`", ty));
        }
    }

    fn emit_err_pat_range(
        &self,
        span: Span,
        lhs: Option<(bool, Ty<'tcx>, Span)>,
        rhs: Option<(bool, Ty<'tcx>, Span)>,
    ) {
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
            "only char and numeric types are allowed in range patterns"
        );
        let msg = |ty| format!("this is of type `{}` but it should be `char` or numeric", ty);
        let mut one_side_err = |first_span, first_ty, second: Option<(bool, Ty<'tcx>, Span)>| {
            err.span_label(first_span, &msg(first_ty));
            if let Some((_, ty, sp)) = second {
                self.endpoint_has_type(&mut err, sp, ty);
            }
        };
        match (lhs, rhs) {
            (Some((true, lhs_ty, lhs_sp)), Some((true, rhs_ty, rhs_sp))) => {
                err.span_label(lhs_sp, &msg(lhs_ty));
                err.span_label(rhs_sp, &msg(rhs_ty));
            }
            (Some((true, lhs_ty, lhs_sp)), rhs) => one_side_err(lhs_sp, lhs_ty, rhs),
            (lhs, Some((true, rhs_ty, rhs_sp))) => one_side_err(rhs_sp, rhs_ty, lhs),
            _ => span_bug!(span, "Impossible, verified above."),
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
        err.emit();
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
            hir::BindingAnnotation::Unannotated => def_bm,
            _ => BindingMode::convert(ba),
        };
        // ...and store it in a side table:
        self.inh.tables.borrow_mut().pat_binding_modes_mut().insert(pat.hir_id, bm);

        debug!("check_pat_ident: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);

        let local_ty = self.local_ty(pat.span, pat.hir_id).decl_ty;
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
            self.check_binding_alt_eq_ty(pat.span, var_id, local_ty, ti);
        }

        if let Some(p) = sub {
            self.check_pat(&p, expected, def_bm, TopInfo { parent_pat: Some(&pat), ..ti });
        }

        local_ty
    }

    fn check_binding_alt_eq_ty(&self, span: Span, var_id: HirId, ty: Ty<'tcx>, ti: TopInfo<'tcx>) {
        let var_ty = self.local_ty(span, var_id).decl_ty;
        if let Some(mut err) = self.demand_eqtype_pat_diag(span, var_ty, ty, ti) {
            let hir = self.tcx.hir();
            let var_ty = self.resolve_vars_with_obligations(var_ty);
            let msg = format!("first introduced with type `{}` here", var_ty);
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
            err.note(&format!("{}a binding must have the same type in all alternatives", pre));
            err.emit();
        }
    }

    fn borrow_pat_suggestion(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        pat: &Pat<'_>,
        inner: &Pat<'_>,
        expected: Ty<'tcx>,
    ) {
        let tcx = self.tcx;
        if let PatKind::Binding(..) = inner.kind {
            let binding_parent_id = tcx.hir().get_parent_node(pat.hir_id);
            let binding_parent = tcx.hir().get(binding_parent_id);
            debug!("inner {:?} pat {:?} parent {:?}", inner, pat, binding_parent);
            match binding_parent {
                hir::Node::Param(hir::Param { span, .. }) => {
                    if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(inner.span) {
                        err.span_suggestion(
                            *span,
                            &format!("did you mean `{}`", snippet),
                            format!(" &{}", expected),
                            Applicability::MachineApplicable,
                        );
                    }
                }
                hir::Node::Arm(_) | hir::Node::Pat(_) => {
                    // rely on match ergonomics or it might be nested `&&pat`
                    if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(inner.span) {
                        err.span_suggestion(
                            pat.span,
                            "you can probably remove the explicit borrow",
                            snippet,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
                _ => {} // don't provide suggestions in other cases #55175
            }
        }
    }

    pub fn check_dereferenceable(&self, span: Span, expected: Ty<'tcx>, inner: &Pat<'_>) -> bool {
        if let PatKind::Binding(..) = inner.kind {
            if let Some(mt) = self.shallow_resolve(expected).builtin_deref(true) {
                if let ty::Dynamic(..) = mt.ty.kind {
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
                    err.span_label(span, format!("type `{}` cannot be dereferenced", type_str));
                    if self.tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ);
                    }
                    err.emit();
                    return false;
                }
            }
        }
        true
    }

    fn check_pat_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &hir::QPath<'_>,
        fields: &'tcx [hir::FieldPat<'tcx>],
        etc: bool,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        // Resolve the path and check the definition for errors.
        let (variant, pat_ty) = if let Some(variant_ty) = self.check_struct_path(qpath, pat.hir_id)
        {
            variant_ty
        } else {
            for field in fields {
                let ti = TopInfo { parent_pat: Some(&pat), ..ti };
                self.check_pat(&field.pat, self.tcx.types.err, def_bm, ti);
            }
            return self.tcx.types.err;
        };

        // Type-check the path.
        self.demand_eqtype_pat(pat.span, expected, pat_ty, ti);

        // Type-check subpatterns.
        if self.check_struct_pat_fields(pat_ty, &pat, variant, fields, etc, def_bm, ti) {
            pat_ty
        } else {
            self.tcx.types.err
        }
    }

    fn check_pat_path(
        &self,
        pat: &Pat<'_>,
        path_resolution: (Res, Option<Ty<'tcx>>, &'b [hir::PathSegment<'b>]),
        qpath: &hir::QPath<'_>,
        expected: Ty<'tcx>,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        // We have already resolved the path.
        let (res, opt_ty, segments) = path_resolution;
        match res {
            Res::Err => {
                self.set_tainted_by_errors();
                return tcx.types.err;
            }
            Res::Def(DefKind::AssocFn, _)
            | Res::Def(DefKind::Ctor(_, CtorKind::Fictive), _)
            | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) => {
                report_unexpected_variant_res(tcx, res, pat.span, qpath);
                return tcx.types.err;
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Const), _)
            | Res::SelfCtor(..)
            | Res::Def(DefKind::Const, _)
            | Res::Def(DefKind::AssocConst, _) => {} // OK
            _ => bug!("unexpected pattern resolution: {:?}", res),
        }

        // Type-check the path.
        let (pat_ty, pat_res) =
            self.instantiate_value_path(segments, opt_ty, res, pat.span, pat.hir_id);
        if let Some(err) =
            self.demand_suptype_with_origin(&self.pattern_cause(ti, pat.span), expected, pat_ty)
        {
            self.emit_bad_pat_path(err, pat.span, res, pat_res, segments, ti.parent_pat);
        }
        pat_ty
    }

    fn emit_bad_pat_path(
        &self,
        mut e: DiagnosticBuilder<'_>,
        pat_span: Span,
        res: Res,
        pat_res: Res,
        segments: &'b [hir::PathSegment<'b>],
        parent_pat: Option<&Pat<'_>>,
    ) {
        if let Some(span) = self.tcx.hir().res_span(pat_res) {
            e.span_label(span, &format!("{} defined here", res.descr()));
            if let [hir::PathSegment { ident, .. }] = &*segments {
                e.span_label(
                    pat_span,
                    &format!(
                        "`{}` is interpreted as {} {}, not a new binding",
                        ident,
                        res.article(),
                        res.descr(),
                    ),
                );
                let (msg, sugg) = match parent_pat {
                    Some(Pat { kind: hir::PatKind::Struct(..), .. }) => (
                        "bind the struct field to a different name instead",
                        format!("{}: other_{}", ident, ident.as_str().to_lowercase()),
                    ),
                    _ => (
                        "introduce a new binding instead",
                        format!("other_{}", ident.as_str().to_lowercase()),
                    ),
                };
                e.span_suggestion(ident.span, msg, sugg, Applicability::HasPlaceholders);
            }
        }
        e.emit();
    }

    fn check_pat_tuple_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &hir::QPath<'_>,
        subpats: &'tcx [&'tcx Pat<'tcx>],
        ddpos: Option<usize>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let on_error = || {
            let parent_pat = Some(pat);
            for pat in subpats {
                self.check_pat(&pat, tcx.types.err, def_bm, TopInfo { parent_pat, ..ti });
            }
        };
        let report_unexpected_res = |res: Res| {
            let msg = format!(
                "expected tuple struct or tuple variant, found {} `{}`",
                res.descr(),
                hir::print::to_string(&tcx.hir(), |s| s.print_qpath(qpath, false)),
            );
            let mut err = struct_span_err!(tcx.sess, pat.span, E0164, "{}", msg);
            match (res, &pat.kind) {
                (Res::Def(DefKind::Fn, _), _) | (Res::Def(DefKind::AssocFn, _), _) => {
                    err.span_label(pat.span, "`fn` calls are not allowed in patterns");
                    err.help(
                        "for more information, visit \
                              https://doc.rust-lang.org/book/ch18-00-patterns.html",
                    );
                }
                _ => {
                    err.span_label(pat.span, "not a tuple variant or struct");
                }
            }
            err.emit();
            on_error();
        };

        // Resolve the path and check the definition for errors.
        let (res, opt_ty, segments) = self.resolve_ty_and_res_ufcs(qpath, pat.hir_id, pat.span);
        if res == Res::Err {
            self.set_tainted_by_errors();
            on_error();
            return self.tcx.types.err;
        }

        // Type-check the path.
        let (pat_ty, res) =
            self.instantiate_value_path(segments, opt_ty, res, pat.span, pat.hir_id);
        if !pat_ty.is_fn() {
            report_unexpected_res(res);
            return tcx.types.err;
        }

        let variant = match res {
            Res::Err => {
                self.set_tainted_by_errors();
                on_error();
                return tcx.types.err;
            }
            Res::Def(DefKind::AssocConst, _) | Res::Def(DefKind::AssocFn, _) => {
                report_unexpected_res(res);
                return tcx.types.err;
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) => tcx.expect_variant_res(res),
            _ => bug!("unexpected pattern resolution: {:?}", res),
        };

        // Replace constructor type with constructed type for tuple struct patterns.
        let pat_ty = pat_ty.fn_sig(tcx).output();
        let pat_ty = pat_ty.no_bound_vars().expect("expected fn type");

        // Type-check the tuple struct pattern against the expected type.
        let diag = self.demand_eqtype_pat_diag(pat.span, expected, pat_ty, ti);
        let had_err = diag.is_some();
        diag.map(|mut err| err.emit());

        // Type-check subpatterns.
        if subpats.len() == variant.fields.len()
            || subpats.len() < variant.fields.len() && ddpos.is_some()
        {
            let substs = match pat_ty.kind {
                ty::Adt(_, substs) => substs,
                _ => bug!("unexpected pattern type {:?}", pat_ty),
            };
            for (i, subpat) in subpats.iter().enumerate_and_adjust(variant.fields.len(), ddpos) {
                let field_ty = self.field_ty(subpat.span, &variant.fields[i], substs);
                self.check_pat(&subpat, field_ty, def_bm, TopInfo { parent_pat: Some(&pat), ..ti });

                self.tcx.check_stability(variant.fields[i].did, Some(pat.hir_id), subpat.span);
            }
        } else {
            // Pattern has wrong number of fields.
            self.e0023(pat.span, res, qpath, subpats, &variant.fields, expected, had_err);
            on_error();
            return tcx.types.err;
        }
        pat_ty
    }

    fn e0023(
        &self,
        pat_span: Span,
        res: Res,
        qpath: &hir::QPath<'_>,
        subpats: &'tcx [&'tcx Pat<'tcx>],
        fields: &'tcx [ty::FieldDef],
        expected: Ty<'tcx>,
        had_err: bool,
    ) {
        let subpats_ending = pluralize!(subpats.len());
        let fields_ending = pluralize!(fields.len());
        let res_span = self.tcx.def_span(res.def_id());
        let mut err = struct_span_err!(
            self.tcx.sess,
            pat_span,
            E0023,
            "this pattern has {} field{}, but the corresponding {} has {} field{}",
            subpats.len(),
            subpats_ending,
            res.descr(),
            fields.len(),
            fields_ending,
        );
        err.span_label(
            pat_span,
            format!("expected {} field{}, found {}", fields.len(), fields_ending, subpats.len(),),
        )
        .span_label(res_span, format!("{} defined here", res.descr()));

        // Identify the case `Some(x, y)` where the expected type is e.g. `Option<(T, U)>`.
        // More generally, the expected type wants a tuple variant with one field of an
        // N-arity-tuple, e.g., `V_i((p_0, .., p_N))`. Meanwhile, the user supplied a pattern
        // with the subpatterns directly in the tuple variant pattern, e.g., `V_i(p_0, .., p_N)`.
        let missing_parenthesis = match (&expected.kind, fields, had_err) {
            // #67037: only do this if we could successfully type-check the expected type against
            // the tuple struct pattern. Otherwise the substs could get out of range on e.g.,
            // `let P() = U;` where `P != U` with `struct P<T>(T);`.
            (ty::Adt(_, substs), [field], false) => {
                let field_ty = self.field_ty(pat_span, field, substs);
                match field_ty.kind {
                    ty::Tuple(_) => field_ty.tuple_fields().count() == subpats.len(),
                    _ => false,
                }
            }
            _ => false,
        };
        if missing_parenthesis {
            let (left, right) = match subpats {
                // This is the zero case; we aim to get the "hi" part of the `QPath`'s
                // span as the "lo" and then the "hi" part of the pattern's span as the "hi".
                // This looks like:
                //
                // help: missing parenthesis
                //   |
                // L |     let A(()) = A(());
                //   |          ^  ^
                [] => {
                    let qpath_span = match qpath {
                        hir::QPath::Resolved(_, path) => path.span,
                        hir::QPath::TypeRelative(_, ps) => ps.ident.span,
                    };
                    (qpath_span.shrink_to_hi(), pat_span)
                }
                // Easy case. Just take the "lo" of the first sub-pattern and the "hi" of the
                // last sub-pattern. In the case of `A(x)` the first and last may coincide.
                // This looks like:
                //
                // help: missing parenthesis
                //   |
                // L |     let A((x, y)) = A((1, 2));
                //   |           ^    ^
                [first, ..] => (first.span.shrink_to_lo(), subpats.last().unwrap().span),
            };
            err.multipart_suggestion(
                "missing parenthesis",
                vec![(left, "(".to_string()), (right.shrink_to_hi(), ")".to_string())],
                Applicability::MachineApplicable,
            );
        }

        err.emit();
    }

    fn check_pat_tuple(
        &self,
        span: Span,
        elements: &'tcx [&'tcx Pat<'tcx>],
        ddpos: Option<usize>,
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let mut expected_len = elements.len();
        if ddpos.is_some() {
            // Require known type only when `..` is present.
            if let ty::Tuple(ref tys) = self.structurally_resolved_type(span, expected).kind {
                expected_len = tys.len();
            }
        }
        let max_len = cmp::max(expected_len, elements.len());

        let element_tys_iter = (0..max_len).map(|_| {
            GenericArg::from(self.next_ty_var(
                // FIXME: `MiscVariable` for now -- obtaining the span and name information
                // from all tuple elements isn't trivial.
                TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span },
            ))
        });
        let element_tys = tcx.mk_substs(element_tys_iter);
        let pat_ty = tcx.mk_ty(ty::Tuple(element_tys));
        if let Some(mut err) = self.demand_eqtype_pat_diag(span, expected, pat_ty, ti) {
            err.emit();
            // Walk subpatterns with an expected type of `err` in this case to silence
            // further errors being emitted when using the bindings. #50333
            let element_tys_iter = (0..max_len).map(|_| tcx.types.err);
            for (_, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.check_pat(elem, &tcx.types.err, def_bm, ti);
            }
            tcx.mk_tup(element_tys_iter)
        } else {
            for (i, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.check_pat(elem, &element_tys[i].expect_ty(), def_bm, ti);
            }
            pat_ty
        }
    }

    fn check_struct_pat_fields(
        &self,
        adt_ty: Ty<'tcx>,
        pat: &'tcx Pat<'tcx>,
        variant: &'tcx ty::VariantDef,
        fields: &'tcx [hir::FieldPat<'tcx>],
        etc: bool,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let (substs, adt) = match adt_ty.kind {
            ty::Adt(adt, substs) => (substs, adt),
            _ => span_bug!(pat.span, "struct pattern is not an ADT"),
        };
        let kind_name = adt.variant_descr();

        // Index the struct fields' types.
        let field_map = variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| (field.ident.modern(), (i, field)))
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
                    self.error_field_already_bound(span, field.ident, *occupied.get());
                    no_field_errors = false;
                    tcx.types.err
                }
                Vacant(vacant) => {
                    vacant.insert(span);
                    field_map
                        .get(&ident)
                        .map(|(i, f)| {
                            self.write_field_index(field.hir_id, *i);
                            self.tcx.check_stability(f.did, Some(pat.hir_id), span);
                            self.field_ty(span, f, substs)
                        })
                        .unwrap_or_else(|| {
                            inexistent_fields.push(field.ident);
                            no_field_errors = false;
                            tcx.types.err
                        })
                }
            };

            self.check_pat(&field.pat, field_ty, def_bm, TopInfo { parent_pat: Some(&pat), ..ti });
        }

        let mut unmentioned_fields = variant
            .fields
            .iter()
            .map(|field| field.ident.modern())
            .filter(|ident| !used_fields.contains_key(&ident))
            .collect::<Vec<_>>();

        if !inexistent_fields.is_empty() && !variant.recovered {
            self.error_inexistent_fields(
                kind_name,
                &inexistent_fields,
                &mut unmentioned_fields,
                variant,
            );
        }

        // Require `..` if struct has non_exhaustive attribute.
        if variant.is_field_list_non_exhaustive() && !adt.did.is_local() && !etc {
            struct_span_err!(
                tcx.sess,
                pat.span,
                E0638,
                "`..` required with {} marked as non-exhaustive",
                kind_name
            )
            .emit();
        }

        // Report an error if incorrect number of the fields were specified.
        if kind_name == "union" {
            if fields.len() != 1 {
                tcx.sess
                    .struct_span_err(pat.span, "union patterns should have exactly one field")
                    .emit();
            }
            if etc {
                tcx.sess.struct_span_err(pat.span, "`..` cannot be used in union patterns").emit();
            }
        } else if !etc && !unmentioned_fields.is_empty() {
            self.error_unmentioned_fields(pat.span, &unmentioned_fields, variant);
        }
        no_field_errors
    }

    fn error_field_already_bound(&self, span: Span, ident: ast::Ident, other_field: Span) {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0025,
            "field `{}` bound multiple times in the pattern",
            ident
        )
        .span_label(span, format!("multiple uses of `{}` in pattern", ident))
        .span_label(other_field, format!("first use of `{}`", ident))
        .emit();
    }

    fn error_inexistent_fields(
        &self,
        kind_name: &str,
        inexistent_fields: &[ast::Ident],
        unmentioned_fields: &mut Vec<ast::Ident>,
        variant: &ty::VariantDef,
    ) {
        let tcx = self.tcx;
        let (field_names, t, plural) = if inexistent_fields.len() == 1 {
            (format!("a field named `{}`", inexistent_fields[0]), "this", "")
        } else {
            (
                format!(
                    "fields named {}",
                    inexistent_fields
                        .iter()
                        .map(|ident| format!("`{}`", ident))
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
                "these",
                "s",
            )
        };
        let spans = inexistent_fields.iter().map(|ident| ident.span).collect::<Vec<_>>();
        let mut err = struct_span_err!(
            tcx.sess,
            spans,
            E0026,
            "{} `{}` does not have {}",
            kind_name,
            tcx.def_path_str(variant.def_id),
            field_names
        );
        if let Some(ident) = inexistent_fields.last() {
            err.span_label(
                ident.span,
                format!(
                    "{} `{}` does not have {} field{}",
                    kind_name,
                    tcx.def_path_str(variant.def_id),
                    t,
                    plural
                ),
            );
            if plural == "" {
                let input = unmentioned_fields.iter().map(|field| &field.name);
                let suggested_name = find_best_match_for_name(input, &ident.as_str(), None);
                if let Some(suggested_name) = suggested_name {
                    err.span_suggestion(
                        ident.span,
                        "a field with a similar name exists",
                        suggested_name.to_string(),
                        Applicability::MaybeIncorrect,
                    );

                    // we don't want to throw `E0027` in case we have thrown `E0026` for them
                    unmentioned_fields.retain(|&x| x.name != suggested_name);
                }
            }
        }
        if tcx.sess.teach(&err.get_code().unwrap()) {
            err.note(
                "This error indicates that a struct pattern attempted to \
                    extract a non-existent field from a struct. Struct fields \
                    are identified by the name used before the colon : so struct \
                    patterns should resemble the declaration of the struct type \
                    being matched.\n\n\
                    If you are using shorthand field patterns but want to refer \
                    to the struct field by a different name, you should rename \
                    it explicitly.",
            );
        }
        err.emit();
    }

    fn error_unmentioned_fields(
        &self,
        span: Span,
        unmentioned_fields: &[ast::Ident],
        variant: &ty::VariantDef,
    ) {
        let field_names = if unmentioned_fields.len() == 1 {
            format!("field `{}`", unmentioned_fields[0])
        } else {
            let fields = unmentioned_fields
                .iter()
                .map(|name| format!("`{}`", name))
                .collect::<Vec<String>>()
                .join(", ");
            format!("fields {}", fields)
        };
        let mut diag = struct_span_err!(
            self.tcx.sess,
            span,
            E0027,
            "pattern does not mention {}",
            field_names
        );
        diag.span_label(span, format!("missing {}", field_names));
        if variant.ctor_kind == CtorKind::Fn {
            diag.note("trying to match a tuple variant with a struct variant pattern");
        }
        if self.tcx.sess.teach(&diag.get_code().unwrap()) {
            diag.note(
                "This error indicates that a pattern for a struct fails to specify a \
                    sub-pattern for every one of the struct's fields. Ensure that each field \
                    from the struct's definition is mentioned in the pattern, or use `..` to \
                    ignore unwanted fields.",
            );
        }
        diag.emit();
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
        let (box_ty, inner_ty) = if self.check_dereferenceable(span, expected, &inner) {
            // Here, `demand::subtype` is good enough, but I don't
            // think any errors can be introduced by using `demand::eqtype`.
            let inner_ty = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span: inner.span,
            });
            let box_ty = tcx.mk_box(inner_ty);
            self.demand_eqtype_pat(span, expected, box_ty, ti);
            (box_ty, inner_ty)
        } else {
            (tcx.types.err, tcx.types.err)
        };
        self.check_pat(&inner, inner_ty, def_bm, ti);
        box_ty
    }

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
        let (rptr_ty, inner_ty) = if self.check_dereferenceable(pat.span, expected, &inner) {
            // `demand::subtype` would be good enough, but using `eqtype` turns
            // out to be equally general. See (note_1) for details.

            // Take region, inner-type from expected type if we can,
            // to avoid creating needless variables. This also helps with
            // the bad  interactions of the given hack detailed in (note_1).
            debug!("check_pat_ref: expected={:?}", expected);
            match expected.kind {
                ty::Ref(_, r_ty, r_mutbl) if r_mutbl == mutbl => (expected, r_ty),
                _ => {
                    let inner_ty = self.next_ty_var(TypeVariableOrigin {
                        kind: TypeVariableOriginKind::TypeInference,
                        span: inner.span,
                    });
                    let rptr_ty = self.new_ref_ty(pat.span, mutbl, inner_ty);
                    debug!("check_pat_ref: demanding {:?} = {:?}", expected, rptr_ty);
                    let err = self.demand_eqtype_pat_diag(pat.span, expected, rptr_ty, ti);

                    // Look for a case like `fn foo(&foo: u32)` and suggest
                    // `fn foo(foo: &u32)`
                    if let Some(mut err) = err {
                        self.borrow_pat_suggestion(&mut err, &pat, &inner, &expected);
                        err.emit();
                    }
                    (rptr_ty, inner_ty)
                }
            }
        } else {
            (tcx.types.err, tcx.types.err)
        };
        self.check_pat(&inner, inner_ty, def_bm, TopInfo { parent_pat: Some(&pat), ..ti });
        rptr_ty
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
    /// ```
    /// [before_0, ..., before_n, (slice, after_0, ... after_n)?]
    /// ```
    /// The type of `slice`, if it is present, depends on the `expected` type.
    /// If `slice` is missing, then so is `after_i`.
    /// If `slice` is present, it can still represent 0 elements.
    fn check_pat_slice(
        &self,
        span: Span,
        before: &'tcx [&'tcx Pat<'tcx>],
        slice: Option<&'tcx Pat<'tcx>>,
        after: &'tcx [&'tcx Pat<'tcx>],
        expected: Ty<'tcx>,
        def_bm: BindingMode,
        ti: TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let err = self.tcx.types.err;
        let expected = self.structurally_resolved_type(span, expected);
        let (inner_ty, slice_ty, expected) = match expected.kind {
            // An array, so we might have something like `let [a, b, c] = [0, 1, 2];`.
            ty::Array(inner_ty, len) => {
                let min = before.len() as u64 + after.len() as u64;
                let slice_ty = self
                    .check_array_pat_len(span, slice, len, min)
                    .map_or(err, |len| self.tcx.mk_array(inner_ty, len));
                (inner_ty, slice_ty, expected)
            }
            ty::Slice(inner_ty) => (inner_ty, expected, expected),
            // The expected type must be an array or slice, but was neither, so error.
            _ => {
                if !expected.references_error() {
                    self.error_expected_array_or_slice(span, expected);
                }
                (err, err, err)
            }
        };

        // Type check all the patterns before `slice`.
        for elt in before {
            self.check_pat(&elt, inner_ty, def_bm, ti);
        }
        // Type check the `slice`, if present, against its expected type.
        if let Some(slice) = slice {
            self.check_pat(&slice, slice_ty, def_bm, ti);
        }
        // Type check the elements after `slice`, if present.
        for elt in after {
            self.check_pat(&elt, inner_ty, def_bm, ti);
        }
        expected
    }

    /// Type check the length of an array pattern.
    ///
    /// Return the length of the variable length pattern,
    /// if it exists and there are no errors.
    fn check_array_pat_len(
        &self,
        span: Span,
        slice: Option<&'tcx Pat<'tcx>>,
        len: &ty::Const<'tcx>,
        min_len: u64,
    ) -> Option<u64> {
        if let Some(len) = len.try_eval_usize(self.tcx, self.param_env) {
            // Now we know the length...
            if slice.is_none() {
                // ...and since there is no variable-length pattern,
                // we require an exact match between the number of elements
                // in the array pattern and as provided by the matched type.
                if min_len != len {
                    self.error_scrutinee_inconsistent_length(span, min_len, len);
                }
            } else if let r @ Some(_) = len.checked_sub(min_len) {
                // The variable-length pattern was there,
                // so it has an array type with the remaining elements left as its size...
                return r;
            } else {
                // ...however, in this case, there were no remaining elements.
                // That is, the slice pattern requires more than the array type offers.
                self.error_scrutinee_with_rest_inconsistent_length(span, min_len, len);
            }
        } else {
            // No idea what the length is, which happens if we have e.g.,
            // `let [a, b] = arr` where `arr: [T; N]` where `const N: usize`.
            self.error_scrutinee_unfixed_length(span);
        }
        None
    }

    fn error_scrutinee_inconsistent_length(&self, span: Span, min_len: u64, size: u64) {
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
        .emit();
    }

    fn error_scrutinee_with_rest_inconsistent_length(&self, span: Span, min_len: u64, size: u64) {
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
        .emit();
    }

    fn error_scrutinee_unfixed_length(&self, span: Span) {
        struct_span_err!(
            self.tcx.sess,
            span,
            E0730,
            "cannot pattern-match on an array without a fixed length",
        )
        .emit();
    }

    fn error_expected_array_or_slice(&self, span: Span, expected_ty: Ty<'tcx>) {
        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0529,
            "expected an array or slice, found `{}`",
            expected_ty
        );
        if let ty::Ref(_, ty, _) = expected_ty.kind {
            if let ty::Array(..) | ty::Slice(..) = ty.kind {
                err.help("the semantics of slice patterns changed recently; see issue #62254");
            }
        }
        err.span_label(span, format!("pattern cannot match with input type `{}`", expected_ty));
        err.emit();
    }
}
