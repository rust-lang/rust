use crate::check::{FnCtxt, Expectation, Diverges, Needs};
use crate::check::coercion::CoerceMany;
use crate::util::nodemap::FxHashMap;
use errors::{Applicability, DiagnosticBuilder};
use rustc::hir::{self, PatKind, Pat, ExprKind};
use rustc::hir::def::{Res, DefKind, CtorKind};
use rustc::hir::pat_util::EnumerateAndAdjustIterator;
use rustc::hir::ptr::P;
use rustc::infer;
use rustc::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc::traits::{ObligationCause, ObligationCauseCode};
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::subst::Kind;
use syntax::ast;
use syntax::source_map::Spanned;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::Span;
use syntax_pos::hygiene::CompilerDesugaringKind;

use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::cmp;

use super::report_unexpected_variant_res;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// `discrim_span` argument having a `Span` indicates that this pattern is part of a match
    /// expression arm guard, and it points to the match discriminant to add context in type errors.
    /// In the following example, `discrim_span` corresponds to the `a + b` expression:
    ///
    /// ```text
    /// error[E0308]: mismatched types
    ///  --> src/main.rs:5:9
    ///   |
    /// 4 |    let temp: usize = match a + b {
    ///   |                            ----- this expression has type `usize`
    /// 5 |         Ok(num) => num,
    ///   |         ^^^^^^^ expected usize, found enum `std::result::Result`
    ///   |
    ///   = note: expected type `usize`
    ///              found type `std::result::Result<_, _>`
    /// ```
    pub fn check_pat_walk(
        &self,
        pat: &'tcx hir::Pat,
        mut expected: Ty<'tcx>,
        mut def_bm: ty::BindingMode,
        discrim_span: Option<Span>,
    ) {
        let tcx = self.tcx;

        debug!("check_pat_walk(pat={:?},expected={:?},def_bm={:?})", pat, expected, def_bm);

        let mut path_resolution = None;
        let is_non_ref_pat = match pat.node {
            PatKind::Struct(..) |
            PatKind::TupleStruct(..) |
            PatKind::Tuple(..) |
            PatKind::Box(_) |
            PatKind::Range(..) |
            PatKind::Slice(..) => true,
            PatKind::Lit(ref lt) => {
                let ty = self.check_expr(lt);
                match ty.sty {
                    ty::Ref(..) => false,
                    _ => true,
                }
            }
            PatKind::Path(ref qpath) => {
                let resolution = self.resolve_ty_and_res_ufcs(qpath, pat.hir_id, pat.span);
                path_resolution = Some(resolution);
                match resolution.0 {
                    Res::Def(DefKind::Const, _) | Res::Def(DefKind::AssocConst, _) => false,
                    _ => true,
                }
            }
            PatKind::Wild |
            PatKind::Binding(..) |
            PatKind::Ref(..) => false,
        };
        if is_non_ref_pat {
            debug!("pattern is non reference pattern");
            let mut exp_ty = self.resolve_type_vars_with_obligations(&expected);

            // Peel off as many `&` or `&mut` from the discriminant as possible. For example,
            // for `match &&&mut Some(5)` the loop runs three times, aborting when it reaches
            // the `Some(5)` which is not of type Ref.
            //
            // For each ampersand peeled off, update the binding mode and push the original
            // type into the adjustments vector.
            //
            // See the examples in `run-pass/match-defbm*.rs`.
            let mut pat_adjustments = vec![];
            while let ty::Ref(_, inner_ty, inner_mutability) = exp_ty.sty {
                debug!("inspecting {:?}", exp_ty);

                debug!("current discriminant is Ref, inserting implicit deref");
                // Preserve the reference type. We'll need it later during HAIR lowering.
                pat_adjustments.push(exp_ty);

                exp_ty = inner_ty;
                def_bm = match def_bm {
                    // If default binding mode is by value, make it `ref` or `ref mut`
                    // (depending on whether we observe `&` or `&mut`).
                    ty::BindByValue(_) =>
                        ty::BindByReference(inner_mutability),

                    // Once a `ref`, always a `ref`. This is because a `& &mut` can't mutate
                    // the underlying value.
                    ty::BindByReference(hir::Mutability::MutImmutable) =>
                        ty::BindByReference(hir::Mutability::MutImmutable),

                    // When `ref mut`, stay a `ref mut` (on `&mut`) or downgrade to `ref`
                    // (on `&`).
                    ty::BindByReference(hir::Mutability::MutMutable) =>
                        ty::BindByReference(inner_mutability),
                };
            }
            expected = exp_ty;

            if pat_adjustments.len() > 0 {
                debug!("default binding mode is now {:?}", def_bm);
                self.inh.tables.borrow_mut()
                    .pat_adjustments_mut()
                    .insert(pat.hir_id, pat_adjustments);
            }
        } else if let PatKind::Ref(..) = pat.node {
            // When you encounter a `&pat` pattern, reset to "by
            // value". This is so that `x` and `y` here are by value,
            // as they appear to be:
            //
            // ```
            // match &(&22, &44) {
            //   (&x, &y) => ...
            // }
            // ```
            //
            // See issue #46688.
            def_bm = ty::BindByValue(hir::MutImmutable);
        }

        // Lose mutability now that we know binding mode and discriminant type.
        let def_bm = def_bm;
        let expected = expected;

        let ty = match pat.node {
            PatKind::Wild => {
                expected
            }
            PatKind::Lit(ref lt) => {
                // We've already computed the type above (when checking for a non-ref pat), so
                // avoid computing it again.
                let ty = self.node_ty(lt.hir_id);

                // Byte string patterns behave the same way as array patterns
                // They can denote both statically and dynamically-sized byte arrays.
                let mut pat_ty = ty;
                if let hir::ExprKind::Lit(ref lt) = lt.node {
                    if let ast::LitKind::ByteStr(_) = lt.node {
                        let expected_ty = self.structurally_resolved_type(pat.span, expected);
                        if let ty::Ref(_, r_ty, _) = expected_ty.sty {
                            if let ty::Slice(_) = r_ty.sty {
                                pat_ty = tcx.mk_imm_ref(tcx.lifetimes.re_static,
                                                        tcx.mk_slice(tcx.types.u8))
                            }
                        }
                    }
                }

                // Somewhat surprising: in this case, the subtyping
                // relation goes the opposite way as the other
                // cases. Actually what we really want is not a subtyping
                // relation at all but rather that there exists a LUB (so
                // that they can be compared). However, in practice,
                // constants are always scalars or strings.  For scalars
                // subtyping is irrelevant, and for strings `ty` is
                // type is `&'static str`, so if we say that
                //
                //     &'static str <: expected
                //
                // then that's equivalent to there existing a LUB.
                if let Some(mut err) = self.demand_suptype_diag(pat.span, expected, pat_ty) {
                    err.emit_unless(discrim_span
                        .filter(|&s| s.is_compiler_desugaring(CompilerDesugaringKind::IfTemporary))
                        .is_some());
                }

                pat_ty
            }
            PatKind::Range(ref begin, ref end, _) => {
                let lhs_ty = self.check_expr(begin);
                let rhs_ty = self.check_expr(end);

                // Check that both end-points are of numeric or char type.
                let numeric_or_char = |ty: Ty<'_>| ty.is_numeric() || ty.is_char();
                let lhs_compat = numeric_or_char(lhs_ty);
                let rhs_compat = numeric_or_char(rhs_ty);

                if !lhs_compat || !rhs_compat {
                    let span = if !lhs_compat && !rhs_compat {
                        pat.span
                    } else if !lhs_compat {
                        begin.span
                    } else {
                        end.span
                    };

                    let mut err = struct_span_err!(
                        tcx.sess,
                        span,
                        E0029,
                        "only char and numeric types are allowed in range patterns"
                    );
                    err.span_label(span, "ranges require char or numeric types");
                    err.note(&format!("start type: {}", self.ty_to_string(lhs_ty)));
                    err.note(&format!("end type: {}", self.ty_to_string(rhs_ty)));
                    if tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(
                            "In a match expression, only numbers and characters can be matched \
                             against a range. This is because the compiler checks that the range \
                             is non-empty at compile-time, and is unable to evaluate arbitrary \
                             comparison functions. If you want to capture values of an orderable \
                             type between two end-points, you can use a guard."
                         );
                    }
                    err.emit();
                    return;
                }

                // Now that we know the types can be unified we find the unified type and use
                // it to type the entire expression.
                let common_type = self.resolve_vars_if_possible(&lhs_ty);

                // Subtyping doesn't matter here, as the value is some kind of scalar.
                self.demand_eqtype_pat(pat.span, expected, lhs_ty, discrim_span);
                self.demand_eqtype_pat(pat.span, expected, rhs_ty, discrim_span);
                common_type
            }
            PatKind::Binding(ba, var_id, _, ref sub) => {
                let bm = if ba == hir::BindingAnnotation::Unannotated {
                    def_bm
                } else {
                    ty::BindingMode::convert(ba)
                };
                self.inh
                    .tables
                    .borrow_mut()
                    .pat_binding_modes_mut()
                    .insert(pat.hir_id, bm);
                debug!("check_pat_walk: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);
                let local_ty = self.local_ty(pat.span, pat.hir_id).decl_ty;
                match bm {
                    ty::BindByReference(mutbl) => {
                        // If the binding is like
                        //     ref x | ref const x | ref mut x
                        // then `x` is assigned a value of type `&M T` where M is the mutability
                        // and T is the expected type.
                        let region_var = self.next_region_var(infer::PatternRegion(pat.span));
                        let mt = ty::TypeAndMut { ty: expected, mutbl: mutbl };
                        let region_ty = tcx.mk_ref(region_var, mt);

                        // `x` is assigned a value of type `&M T`, hence `&M T <: typeof(x)` is
                        // required. However, we use equality, which is stronger. See (*) for
                        // an explanation.
                        self.demand_eqtype_pat(pat.span, region_ty, local_ty, discrim_span);
                    }
                    // Otherwise, the type of x is the expected type `T`.
                    ty::BindByValue(_) => {
                        // As above, `T <: typeof(x)` is required, but we
                        // use equality, see (*) below.
                        self.demand_eqtype_pat(pat.span, expected, local_ty, discrim_span);
                    }
                }

                // If there are multiple arms, make sure they all agree on
                // what the type of the binding `x` ought to be.
                if var_id != pat.hir_id {
                    let vt = self.local_ty(pat.span, var_id).decl_ty;
                    self.demand_eqtype_pat(pat.span, vt, local_ty, discrim_span);
                }

                if let Some(ref p) = *sub {
                    self.check_pat_walk(&p, expected, def_bm, discrim_span);
                }

                local_ty
            }
            PatKind::TupleStruct(ref qpath, ref subpats, ddpos) => {
                self.check_pat_tuple_struct(
                    pat,
                    qpath,
                    &subpats,
                    ddpos,
                    expected,
                    def_bm,
                    discrim_span,
                )
            }
            PatKind::Path(ref qpath) => {
                self.check_pat_path(pat, path_resolution.unwrap(), qpath, expected)
            }
            PatKind::Struct(ref qpath, ref fields, etc) => {
                self.check_pat_struct(pat, qpath, fields, etc, expected, def_bm, discrim_span)
            }
            PatKind::Tuple(ref elements, ddpos) => {
                let mut expected_len = elements.len();
                if ddpos.is_some() {
                    // Require known type only when `..` is present.
                    if let ty::Tuple(ref tys) =
                            self.structurally_resolved_type(pat.span, expected).sty {
                        expected_len = tys.len();
                    }
                }
                let max_len = cmp::max(expected_len, elements.len());

                let element_tys_iter = (0..max_len).map(|_| {
                    Kind::from(self.next_ty_var(
                        // FIXME: `MiscVariable` for now -- obtaining the span and name information
                        // from all tuple elements isn't trivial.
                        TypeVariableOrigin {
                            kind: TypeVariableOriginKind::TypeInference,
                            span: pat.span,
                        },
                    ))
                });
                let element_tys = tcx.mk_substs(element_tys_iter);
                let pat_ty = tcx.mk_ty(ty::Tuple(element_tys));
                if let Some(mut err) = self.demand_eqtype_diag(pat.span, expected, pat_ty) {
                    err.emit();
                    // Walk subpatterns with an expected type of `err` in this case to silence
                    // further errors being emitted when using the bindings. #50333
                    let element_tys_iter = (0..max_len).map(|_| tcx.types.err);
                    for (_, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                        self.check_pat_walk(elem, &tcx.types.err, def_bm, discrim_span);
                    }
                    tcx.mk_tup(element_tys_iter)
                } else {
                    for (i, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                        self.check_pat_walk(
                            elem,
                            &element_tys[i].expect_ty(),
                            def_bm,
                            discrim_span,
                        );
                    }
                    pat_ty
                }
            }
            PatKind::Box(ref inner) => {
                let inner_ty = self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeInference,
                    span: inner.span,
                });
                let uniq_ty = tcx.mk_box(inner_ty);

                if self.check_dereferencable(pat.span, expected, &inner) {
                    // Here, `demand::subtype` is good enough, but I don't
                    // think any errors can be introduced by using
                    // `demand::eqtype`.
                    self.demand_eqtype_pat(pat.span, expected, uniq_ty, discrim_span);
                    self.check_pat_walk(&inner, inner_ty, def_bm, discrim_span);
                    uniq_ty
                } else {
                    self.check_pat_walk(&inner, tcx.types.err, def_bm, discrim_span);
                    tcx.types.err
                }
            }
            PatKind::Ref(ref inner, mutbl) => {
                let expected = self.shallow_resolve(expected);
                if self.check_dereferencable(pat.span, expected, &inner) {
                    // `demand::subtype` would be good enough, but using
                    // `eqtype` turns out to be equally general. See (*)
                    // below for details.

                    // Take region, inner-type from expected type if we
                    // can, to avoid creating needless variables.  This
                    // also helps with the bad interactions of the given
                    // hack detailed in (*) below.
                    debug!("check_pat_walk: expected={:?}", expected);
                    let (rptr_ty, inner_ty) = match expected.sty {
                        ty::Ref(_, r_ty, r_mutbl) if r_mutbl == mutbl => {
                            (expected, r_ty)
                        }
                        _ => {
                            let inner_ty = self.next_ty_var(
                                TypeVariableOrigin {
                                    kind: TypeVariableOriginKind::TypeInference,
                                    span: inner.span,
                                }
                            );
                            let mt = ty::TypeAndMut { ty: inner_ty, mutbl: mutbl };
                            let region = self.next_region_var(infer::PatternRegion(pat.span));
                            let rptr_ty = tcx.mk_ref(region, mt);
                            debug!("check_pat_walk: demanding {:?} = {:?}", expected, rptr_ty);
                            let err = self.demand_eqtype_diag(pat.span, expected, rptr_ty);

                            // Look for a case like `fn foo(&foo: u32)` and suggest
                            // `fn foo(foo: &u32)`
                            if let Some(mut err) = err {
                                self.borrow_pat_suggestion(&mut err, &pat, &inner, &expected);
                                err.emit();
                            }
                            (rptr_ty, inner_ty)
                        }
                    };

                    self.check_pat_walk(&inner, inner_ty, def_bm, discrim_span);
                    rptr_ty
                } else {
                    self.check_pat_walk(&inner, tcx.types.err, def_bm, discrim_span);
                    tcx.types.err
                }
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                let expected_ty = self.structurally_resolved_type(pat.span, expected);
                let (inner_ty, slice_ty) = match expected_ty.sty {
                    ty::Array(inner_ty, size) => {
                        if let Some(size) = size.assert_usize(tcx) {
                            let min_len = before.len() as u64 + after.len() as u64;
                            if slice.is_none() {
                                if min_len != size {
                                    struct_span_err!(
                                        tcx.sess, pat.span, E0527,
                                        "pattern requires {} elements but array has {}",
                                        min_len, size)
                                        .span_label(pat.span, format!("expected {} elements", size))
                                        .emit();
                                }
                                (inner_ty, tcx.types.err)
                            } else if let Some(rest) = size.checked_sub(min_len) {
                                (inner_ty, tcx.mk_array(inner_ty, rest))
                            } else {
                                struct_span_err!(tcx.sess, pat.span, E0528,
                                        "pattern requires at least {} elements but array has {}",
                                        min_len, size)
                                    .span_label(pat.span,
                                        format!("pattern cannot match array of {} elements", size))
                                    .emit();
                                (inner_ty, tcx.types.err)
                            }
                        } else {
                            struct_span_err!(
                                tcx.sess,
                                pat.span,
                                E0730,
                                "cannot pattern-match on an array without a fixed length",
                            ).emit();
                            (inner_ty, tcx.types.err)
                        }
                    }
                    ty::Slice(inner_ty) => (inner_ty, expected_ty),
                    _ => {
                        if !expected_ty.references_error() {
                            let mut err = struct_span_err!(
                                tcx.sess, pat.span, E0529,
                                "expected an array or slice, found `{}`",
                                expected_ty);
                            if let ty::Ref(_, ty, _) = expected_ty.sty {
                                match ty.sty {
                                    ty::Array(..) | ty::Slice(..) => {
                                        err.help("the semantics of slice patterns changed \
                                                  recently; see issue #62254");
                                    }
                                    _ => {}
                                }
                            }

                            err.span_label( pat.span,
                                format!("pattern cannot match with input type `{}`", expected_ty)
                            ).emit();
                        }
                        (tcx.types.err, tcx.types.err)
                    }
                };

                for elt in before {
                    self.check_pat_walk(&elt, inner_ty, def_bm, discrim_span);
                }
                if let Some(ref slice) = *slice {
                    self.check_pat_walk(&slice, slice_ty, def_bm, discrim_span);
                }
                for elt in after {
                    self.check_pat_walk(&elt, inner_ty, def_bm, discrim_span);
                }
                expected_ty
            }
        };

        self.write_ty(pat.hir_id, ty);

        // (*) In most of the cases above (literals and constants being
        // the exception), we relate types using strict equality, even
        // though subtyping would be sufficient. There are a few reasons
        // for this, some of which are fairly subtle and which cost me
        // (nmatsakis) an hour or two debugging to remember, so I thought
        // I'd write them down this time.
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

    fn borrow_pat_suggestion(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        pat: &Pat,
        inner: &Pat,
        expected: Ty<'tcx>,
    ) {
        let tcx = self.tcx;
        if let PatKind::Binding(..) = inner.node {
            let parent_id = tcx.hir().get_parent_node(pat.hir_id);
            let parent = tcx.hir().get(parent_id);
            debug!("inner {:?} pat {:?} parent {:?}", inner, pat, parent);
            match parent {
                hir::Node::Item(hir::Item { node: hir::ItemKind::Fn(..), .. }) |
                hir::Node::ForeignItem(hir::ForeignItem {
                    node: hir::ForeignItemKind::Fn(..), ..
                }) |
                hir::Node::TraitItem(hir::TraitItem { node: hir::TraitItemKind::Method(..), .. }) |
                hir::Node::ImplItem(hir::ImplItem { node: hir::ImplItemKind::Method(..), .. }) => {
                    // this pat is likely an argument
                    if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(inner.span) {
                        // FIXME: turn into structured suggestion, will need a span that also
                        // includes the the arg's type.
                        err.help(&format!("did you mean `{}: &{}`?", snippet, expected));
                    }
                }
                hir::Node::Arm(_) |
                hir::Node::Pat(_) => {
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

    pub fn check_dereferencable(&self, span: Span, expected: Ty<'tcx>, inner: &hir::Pat) -> bool {
        if let PatKind::Binding(..) = inner.node {
            if let Some(mt) = self.shallow_resolve(expected).builtin_deref(true) {
                if let ty::Dynamic(..) = mt.ty.sty {
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
                        err.note("\
This error indicates that a pointer to a trait type cannot be implicitly dereferenced by a \
pattern. Every trait defines a type, but because the size of trait implementors isn't fixed, \
this type has no compile-time size. Therefore, all accesses to trait types must be through \
pointers. If you encounter this error you should try to avoid dereferencing the pointer.

You can read more about trait objects in the Trait Objects section of the Reference: \
https://doc.rust-lang.org/reference/types.html#trait-objects");
                    }
                    err.emit();
                    return false
                }
            }
        }
        true
    }

    pub fn check_match(
        &self,
        expr: &'tcx hir::Expr,
        discrim: &'tcx hir::Expr,
        arms: &'tcx [hir::Arm],
        expected: Expectation<'tcx>,
        match_src: hir::MatchSource,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        use hir::MatchSource::*;
        let (source_if, if_no_else, if_desugar) = match match_src {
            IfDesugar { contains_else_clause } => (true, !contains_else_clause, true),
            IfLetDesugar { contains_else_clause } => (true, !contains_else_clause, false),
            _ => (false, false, false),
        };

        // Type check the descriminant and get its type.
        let discrim_ty = if if_desugar {
            // Here we want to ensure:
            //
            // 1. That default match bindings are *not* accepted in the condition of an
            //    `if` expression. E.g. given `fn foo() -> &bool;` we reject `if foo() { .. }`.
            //
            // 2. By expecting `bool` for `expr` we get nice diagnostics for e.g. `if x = y { .. }`.
            //
            // FIXME(60707): Consider removing hack with principled solution.
            self.check_expr_has_type_or_error(discrim, self.tcx.types.bool)
        } else {
            self.demand_discriminant_type(arms, discrim)
        };

        // If there are no arms, that is a diverging match; a special case.
        if arms.is_empty() {
            self.diverges.set(self.diverges.get() | Diverges::Always);
            return tcx.types.never;
        }

        self.warn_arms_when_scrutinee_diverges(arms, source_if);

        // Otherwise, we have to union together the types that the
        // arms produce and so forth.
        let discrim_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        // rust-lang/rust#55810: Typecheck patterns first (via eager
        // collection into `Vec`), so we get types for all bindings.
        let all_arm_pats_diverge: Vec<_> = arms.iter().map(|arm| {
            let mut all_pats_diverge = Diverges::WarnedAlways;
            for p in &arm.pats {
                self.diverges.set(Diverges::Maybe);
                let binding_mode = ty::BindingMode::BindByValue(hir::Mutability::MutImmutable);
                self.check_pat_walk(&p, discrim_ty, binding_mode, Some(discrim.span));
                all_pats_diverge &= self.diverges.get();
            }

            // As discussed with @eddyb, this is for disabling unreachable_code
            // warnings on patterns (they're now subsumed by unreachable_patterns
            // warnings).
            match all_pats_diverge {
                Diverges::Maybe => Diverges::Maybe,
                Diverges::Always | Diverges::WarnedAlways => Diverges::WarnedAlways,
            }
        }).collect();

        // Now typecheck the blocks.
        //
        // The result of the match is the common supertype of all the
        // arms. Start out the value as bottom, since it's the, well,
        // bottom the type lattice, and we'll be moving up the lattice as
        // we process each arm. (Note that any match with 0 arms is matching
        // on any empty type and is therefore unreachable; should the flow
        // of execution reach it, we will panic, so bottom is an appropriate
        // type in that case)
        let mut all_arms_diverge = Diverges::WarnedAlways;

        let expected = expected.adjust_for_branches(self);

        let mut coercion = {
            let coerce_first = match expected {
                // We don't coerce to `()` so that if the match expression is a
                // statement it's branches can have any consistent type. That allows
                // us to give better error messages (pointing to a usually better
                // arm for inconsistent arms or to the whole match when a `()` type
                // is required).
                Expectation::ExpectHasType(ety) if ety != self.tcx.mk_unit() => ety,
                _ => self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: expr.span,
                }),
            };
            CoerceMany::with_coercion_sites(coerce_first, arms)
        };

        let mut other_arms = vec![];  // used only for diagnostics
        let mut prior_arm_ty = None;
        for (i, (arm, pats_diverge)) in arms.iter().zip(all_arm_pats_diverge).enumerate() {
            if let Some(g) = &arm.guard {
                self.diverges.set(pats_diverge);
                match g {
                    hir::Guard::If(e) => self.check_expr_has_type_or_error(e, tcx.types.bool),
                };
            }

            self.diverges.set(pats_diverge);
            let arm_ty = self.check_expr_with_expectation(&arm.body, expected);
            all_arms_diverge &= self.diverges.get();

            let span = expr.span;

            if source_if {
                let then_expr = &arms[0].body;
                match (i, if_no_else) {
                    (0, _) => coercion.coerce(self, &self.misc(span), then_expr, arm_ty),
                    (_, true) => self.if_fallback_coercion(span, then_expr, &mut coercion),
                    (_, _) => {
                        let then_ty = prior_arm_ty.unwrap();
                        let cause = self.if_cause(span, then_expr, &arm.body, then_ty, arm_ty);
                        coercion.coerce(self, &cause, &arm.body, arm_ty);
                    }
                }
            } else {
                let arm_span = if let hir::ExprKind::Block(blk, _) = &arm.body.node {
                    // Point at the block expr instead of the entire block
                    blk.expr.as_ref().map(|e| e.span).unwrap_or(arm.body.span)
                } else {
                    arm.body.span
                };
                let (span, code) = match i {
                    // The reason for the first arm to fail is not that the match arms diverge,
                    // but rather that there's a prior obligation that doesn't hold.
                    0 => (arm_span, ObligationCauseCode::BlockTailExpression(arm.body.hir_id)),
                    _ => (span, ObligationCauseCode::MatchExpressionArm {
                        arm_span,
                        source: match_src,
                        prior_arms: other_arms.clone(),
                        last_ty: prior_arm_ty.unwrap(),
                        discrim_hir_id: discrim.hir_id,
                    }),
                };
                let cause = self.cause(span, code);
                coercion.coerce(self, &cause, &arm.body, arm_ty);
                other_arms.push(arm_span);
                if other_arms.len() > 5 {
                    other_arms.remove(0);
                }
            }
            prior_arm_ty = Some(arm_ty);
        }

        // We won't diverge unless the discriminant or all arms diverge.
        self.diverges.set(discrim_diverges | all_arms_diverge);

        coercion.complete(self)
    }

    /// When the previously checked expression (the scrutinee) diverges,
    /// warn the user about the match arms being unreachable.
    fn warn_arms_when_scrutinee_diverges(&self, arms: &'tcx [hir::Arm], source_if: bool) {
        if self.diverges.get().always() {
            let msg = if source_if { "block in `if` expression" } else { "arm" };
            for arm in arms {
                self.warn_if_unreachable(arm.body.hir_id, arm.body.span, msg);
            }
        }
    }

    /// Handle the fallback arm of a desugared if(-let) like a missing else.
    fn if_fallback_coercion(
        &self,
        span: Span,
        then_expr: &'tcx hir::Expr,
        coercion: &mut CoerceMany<'tcx, '_, rustc::hir::Arm>,
    ) {
        // If this `if` expr is the parent's function return expr,
        // the cause of the type coercion is the return type, point at it. (#25228)
        let ret_reason = self.maybe_get_coercion_reason(then_expr.hir_id, span);
        let cause = self.cause(span, ObligationCauseCode::IfExpressionWithNoElse);
        coercion.coerce_forced_unit(self, &cause, &mut |err| {
            if let Some((span, msg)) = &ret_reason {
                err.span_label(*span, msg.as_str());
            } else if let ExprKind::Block(block, _) = &then_expr.node {
                if let Some(expr) = &block.expr {
                    err.span_label(expr.span, "found here".to_string());
                }
            }
            err.note("`if` expressions without `else` evaluate to `()`");
            err.help("consider adding an `else` block that evaluates to the expected type");
        }, ret_reason.is_none());
    }

    fn maybe_get_coercion_reason(&self, hir_id: hir::HirId, span: Span) -> Option<(Span, String)> {
        use hir::Node::{Block, Item, Local};

        let hir = self.tcx.hir();
        let arm_id = hir.get_parent_node(hir_id);
        let match_id = hir.get_parent_node(arm_id);
        let containing_id = hir.get_parent_node(match_id);

        let node = hir.get(containing_id);
        if let Block(block) = node {
            // check that the body's parent is an fn
            let parent = hir.get(
                hir.get_parent_node(
                    hir.get_parent_node(block.hir_id),
                ),
            );
            if let (Some(expr), Item(hir::Item {
                node: hir::ItemKind::Fn(..), ..
            })) = (&block.expr, parent) {
                // check that the `if` expr without `else` is the fn body's expr
                if expr.span == span {
                    return self.get_fn_decl(hir_id).map(|(fn_decl, _)| (
                        fn_decl.output.span(),
                        format!("expected `{}` because of this return type", fn_decl.output),
                    ));
                }
            }
        }
        if let Local(hir::Local { ty: Some(_), pat, .. }) = node {
            return Some((pat.span, "expected because of this assignment".to_string()));
        }
        None
    }

    fn if_cause(
        &self,
        span: Span,
        then_expr: &'tcx hir::Expr,
        else_expr: &'tcx hir::Expr,
        then_ty: Ty<'tcx>,
        else_ty: Ty<'tcx>,
    ) -> ObligationCause<'tcx> {
        let mut outer_sp = if self.tcx.sess.source_map().is_multiline(span) {
            // The `if`/`else` isn't in one line in the output, include some context to make it
            // clear it is an if/else expression:
            // ```
            // LL |      let x = if true {
            //    | _____________-
            // LL ||         10i32
            //    ||         ----- expected because of this
            // LL ||     } else {
            // LL ||         10u32
            //    ||         ^^^^^ expected i32, found u32
            // LL ||     };
            //    ||_____- if and else have incompatible types
            // ```
            Some(span)
        } else {
            // The entire expression is in one line, only point at the arms
            // ```
            // LL |     let x = if true { 10i32 } else { 10u32 };
            //    |                       -----          ^^^^^ expected i32, found u32
            //    |                       |
            //    |                       expected because of this
            // ```
            None
        };

        let mut remove_semicolon = None;
        let error_sp = if let ExprKind::Block(block, _) = &else_expr.node {
            if let Some(expr) = &block.expr {
                expr.span
            } else if let Some(stmt) = block.stmts.last() {
                // possibly incorrect trailing `;` in the else arm
                remove_semicolon = self.could_remove_semicolon(block, then_ty);
                stmt.span
            } else { // empty block; point at its entirety
                // Avoid overlapping spans that aren't as readable:
                // ```
                // 2 |        let x = if true {
                //   |   _____________-
                // 3 |  |         3
                //   |  |         - expected because of this
                // 4 |  |     } else {
                //   |  |____________^
                // 5 | ||
                // 6 | ||     };
                //   | ||     ^
                //   | ||_____|
                //   | |______if and else have incompatible types
                //   |        expected integer, found ()
                // ```
                // by not pointing at the entire expression:
                // ```
                // 2 |       let x = if true {
                //   |               ------- if and else have incompatible types
                // 3 |           3
                //   |           - expected because of this
                // 4 |       } else {
                //   |  ____________^
                // 5 | |
                // 6 | |     };
                //   | |_____^ expected integer, found ()
                // ```
                if outer_sp.is_some() {
                    outer_sp = Some(self.tcx.sess.source_map().def_span(span));
                }
                else_expr.span
            }
        } else { // shouldn't happen unless the parser has done something weird
            else_expr.span
        };

        // Compute `Span` of `then` part of `if`-expression.
        let then_sp = if let ExprKind::Block(block, _) = &then_expr.node {
            if let Some(expr) = &block.expr {
                expr.span
            } else if let Some(stmt) = block.stmts.last() {
                // possibly incorrect trailing `;` in the else arm
                remove_semicolon = remove_semicolon.or(self.could_remove_semicolon(block, else_ty));
                stmt.span
            } else { // empty block; point at its entirety
                outer_sp = None;  // same as in `error_sp`; cleanup output
                then_expr.span
            }
        } else { // shouldn't happen unless the parser has done something weird
            then_expr.span
        };

        // Finally construct the cause:
        self.cause(error_sp, ObligationCauseCode::IfExpression {
            then: then_sp,
            outer: outer_sp,
            semicolon: remove_semicolon,
        })
    }

    fn demand_discriminant_type(
        &self,
        arms: &'tcx [hir::Arm],
        discrim: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        // Not entirely obvious: if matches may create ref bindings, we want to
        // use the *precise* type of the discriminant, *not* some supertype, as
        // the "discriminant type" (issue #23116).
        //
        // arielb1 [writes here in this comment thread][c] that there
        // is certainly *some* potential danger, e.g., for an example
        // like:
        //
        // [c]: https://github.com/rust-lang/rust/pull/43399#discussion_r130223956
        //
        // ```
        // let Foo(x) = f()[0];
        // ```
        //
        // Then if the pattern matches by reference, we want to match
        // `f()[0]` as a lexpr, so we can't allow it to be
        // coerced. But if the pattern matches by value, `f()[0]` is
        // still syntactically a lexpr, but we *do* want to allow
        // coercions.
        //
        // However, *likely* we are ok with allowing coercions to
        // happen if there are no explicit ref mut patterns - all
        // implicit ref mut patterns must occur behind a reference, so
        // they will have the "correct" variance and lifetime.
        //
        // This does mean that the following pattern would be legal:
        //
        // ```
        // struct Foo(Bar);
        // struct Bar(u32);
        // impl Deref for Foo {
        //     type Target = Bar;
        //     fn deref(&self) -> &Bar { &self.0 }
        // }
        // impl DerefMut for Foo {
        //     fn deref_mut(&mut self) -> &mut Bar { &mut self.0 }
        // }
        // fn foo(x: &mut Foo) {
        //     {
        //         let Bar(z): &mut Bar = x;
        //         *z = 42;
        //     }
        //     assert_eq!(foo.0.0, 42);
        // }
        // ```
        //
        // FIXME(tschottdorf): don't call contains_explicit_ref_binding, which
        // is problematic as the HIR is being scraped, but ref bindings may be
        // implicit after #42640. We need to make sure that pat_adjustments
        // (once introduced) is populated by the time we get here.
        //
        // See #44848.
        let contains_ref_bindings = arms.iter()
                                        .filter_map(|a| a.contains_explicit_ref_binding())
                                        .max_by_key(|m| match *m {
                                            hir::MutMutable => 1,
                                            hir::MutImmutable => 0,
                                        });

        if let Some(m) = contains_ref_bindings {
            self.check_expr_with_needs(discrim, Needs::maybe_mut_place(m))
        } else {
            // ...but otherwise we want to use any supertype of the
            // discriminant. This is sort of a workaround, see note (*) in
            // `check_pat` for some details.
            let discrim_ty = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span: discrim.span,
            });
            self.check_expr_has_type_or_error(discrim, discrim_ty);
            discrim_ty
        }
    }

    fn check_pat_struct(
        &self,
        pat: &'tcx hir::Pat,
        qpath: &hir::QPath,
        fields: &'tcx [Spanned<hir::FieldPat>],
        etc: bool,
        expected: Ty<'tcx>,
        def_bm: ty::BindingMode,
        discrim_span: Option<Span>,
    ) -> Ty<'tcx> {
        // Resolve the path and check the definition for errors.
        let (variant, pat_ty) = if let Some(variant_ty) = self.check_struct_path(qpath, pat.hir_id)
        {
            variant_ty
        } else {
            for field in fields {
                self.check_pat_walk(&field.node.pat, self.tcx.types.err, def_bm, discrim_span);
            }
            return self.tcx.types.err;
        };

        // Type-check the path.
        self.demand_eqtype_pat(pat.span, expected, pat_ty, discrim_span);

        // Type-check subpatterns.
        if self.check_struct_pat_fields(pat_ty, pat.hir_id, pat.span, variant, fields, etc, def_bm)
        {
            pat_ty
        } else {
            self.tcx.types.err
        }
    }

    fn check_pat_path(
        &self,
        pat: &hir::Pat,
        path_resolution: (Res, Option<Ty<'tcx>>, &'b [hir::PathSegment]),
        qpath: &hir::QPath,
        expected: Ty<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        // We have already resolved the path.
        let (res, opt_ty, segments) = path_resolution;
        match res {
            Res::Err => {
                self.set_tainted_by_errors();
                return tcx.types.err;
            }
            Res::Def(DefKind::Method, _) |
            Res::Def(DefKind::Ctor(_, CtorKind::Fictive), _) |
            Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) => {
                report_unexpected_variant_res(tcx, res, pat.span, qpath);
                return tcx.types.err;
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Const), _) | Res::SelfCtor(..) |
            Res::Def(DefKind::Const, _) | Res::Def(DefKind::AssocConst, _) => {} // OK
            _ => bug!("unexpected pattern resolution: {:?}", res)
        }

        // Type-check the path.
        let pat_ty = self.instantiate_value_path(segments, opt_ty, res, pat.span, pat.hir_id).0;
        self.demand_suptype(pat.span, expected, pat_ty);
        pat_ty
    }

    fn check_pat_tuple_struct(
        &self,
        pat: &hir::Pat,
        qpath: &hir::QPath,
        subpats: &'tcx [P<hir::Pat>],
        ddpos: Option<usize>,
        expected: Ty<'tcx>,
        def_bm: ty::BindingMode,
        match_arm_pat_span: Option<Span>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let on_error = || {
            for pat in subpats {
                self.check_pat_walk(&pat, tcx.types.err, def_bm, match_arm_pat_span);
            }
        };
        let report_unexpected_res = |res: Res| {
            let msg = format!("expected tuple struct/variant, found {} `{}`",
                              res.descr(),
                              hir::print::to_string(tcx.hir(), |s| s.print_qpath(qpath, false)));
            let mut err = struct_span_err!(tcx.sess, pat.span, E0164, "{}", msg);
            match (res, &pat.node) {
                (Res::Def(DefKind::Fn, _), _) | (Res::Def(DefKind::Method, _), _) => {
                    err.span_label(pat.span, "`fn` calls are not allowed in patterns");
                    err.help("for more information, visit \
                              https://doc.rust-lang.org/book/ch18-00-patterns.html");
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
        let (pat_ty, res) = self.instantiate_value_path(segments, opt_ty, res, pat.span,
            pat.hir_id);
        if !pat_ty.is_fn() {
            report_unexpected_res(res);
            return self.tcx.types.err;
        }

        let variant = match res {
            Res::Err => {
                self.set_tainted_by_errors();
                on_error();
                return tcx.types.err;
            }
            Res::Def(DefKind::AssocConst, _) | Res::Def(DefKind::Method, _) => {
                report_unexpected_res(res);
                return tcx.types.err;
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) => {
                tcx.expect_variant_res(res)
            }
            _ => bug!("unexpected pattern resolution: {:?}", res)
        };

        // Replace constructor type with constructed type for tuple struct patterns.
        let pat_ty = pat_ty.fn_sig(tcx).output();
        let pat_ty = pat_ty.no_bound_vars().expect("expected fn type");

        self.demand_eqtype_pat(pat.span, expected, pat_ty, match_arm_pat_span);

        // Type-check subpatterns.
        if subpats.len() == variant.fields.len() ||
                subpats.len() < variant.fields.len() && ddpos.is_some() {
            let substs = match pat_ty.sty {
                ty::Adt(_, substs) => substs,
                _ => bug!("unexpected pattern type {:?}", pat_ty),
            };
            for (i, subpat) in subpats.iter().enumerate_and_adjust(variant.fields.len(), ddpos) {
                let field_ty = self.field_ty(subpat.span, &variant.fields[i], substs);
                self.check_pat_walk(&subpat, field_ty, def_bm, match_arm_pat_span);

                self.tcx.check_stability(variant.fields[i].did, Some(pat.hir_id), subpat.span);
            }
        } else {
            let subpats_ending = if subpats.len() == 1 { "" } else { "s" };
            let fields_ending = if variant.fields.len() == 1 { "" } else { "s" };
            struct_span_err!(tcx.sess, pat.span, E0023,
                             "this pattern has {} field{}, but the corresponding {} has {} field{}",
                             subpats.len(), subpats_ending, res.descr(),
                             variant.fields.len(),  fields_ending)
                .span_label(pat.span, format!("expected {} field{}, found {}",
                                              variant.fields.len(), fields_ending, subpats.len()))
                .emit();
            on_error();
            return tcx.types.err;
        }
        pat_ty
    }

    fn check_struct_pat_fields(
        &self,
        adt_ty: Ty<'tcx>,
        pat_id: hir::HirId,
        span: Span,
        variant: &'tcx ty::VariantDef,
        fields: &'tcx [Spanned<hir::FieldPat>],
        etc: bool,
        def_bm: ty::BindingMode,
    ) -> bool {
        let tcx = self.tcx;

        let (substs, adt) = match adt_ty.sty {
            ty::Adt(adt, substs) => (substs, adt),
            _ => span_bug!(span, "struct pattern is not an ADT")
        };
        let kind_name = adt.variant_descr();

        // Index the struct fields' types.
        let field_map = variant.fields
            .iter()
            .enumerate()
            .map(|(i, field)| (field.ident.modern(), (i, field)))
            .collect::<FxHashMap<_, _>>();

        // Keep track of which fields have already appeared in the pattern.
        let mut used_fields = FxHashMap::default();
        let mut no_field_errors = true;

        let mut inexistent_fields = vec![];
        // Typecheck each field.
        for &Spanned { node: ref field, span } in fields {
            let ident = tcx.adjust_ident(field.ident, variant.def_id);
            let field_ty = match used_fields.entry(ident) {
                Occupied(occupied) => {
                    struct_span_err!(tcx.sess, span, E0025,
                                     "field `{}` bound multiple times \
                                      in the pattern",
                                     field.ident)
                        .span_label(span,
                                    format!("multiple uses of `{}` in pattern", field.ident))
                        .span_label(*occupied.get(), format!("first use of `{}`", field.ident))
                        .emit();
                    no_field_errors = false;
                    tcx.types.err
                }
                Vacant(vacant) => {
                    vacant.insert(span);
                    field_map.get(&ident)
                        .map(|(i, f)| {
                            self.write_field_index(field.hir_id, *i);
                            self.tcx.check_stability(f.did, Some(pat_id), span);
                            self.field_ty(span, f, substs)
                        })
                        .unwrap_or_else(|| {
                            inexistent_fields.push(field.ident);
                            no_field_errors = false;
                            tcx.types.err
                        })
                }
            };

            self.check_pat_walk(&field.pat, field_ty, def_bm, None);
        }
        let mut unmentioned_fields = variant.fields
                .iter()
                .map(|field| field.ident.modern())
                .filter(|ident| !used_fields.contains_key(&ident))
                .collect::<Vec<_>>();
        if inexistent_fields.len() > 0 && !variant.recovered {
            let (field_names, t, plural) = if inexistent_fields.len() == 1 {
                (format!("a field named `{}`", inexistent_fields[0]), "this", "")
            } else {
                (format!("fields named {}",
                         inexistent_fields.iter()
                            .map(|ident| format!("`{}`", ident))
                            .collect::<Vec<String>>()
                            .join(", ")), "these", "s")
            };
            let spans = inexistent_fields.iter().map(|ident| ident.span).collect::<Vec<_>>();
            let mut err = struct_span_err!(tcx.sess,
                                           spans,
                                           E0026,
                                           "{} `{}` does not have {}",
                                           kind_name,
                                           tcx.def_path_str(variant.def_id),
                                           field_names);
            if let Some(ident) = inexistent_fields.last() {
                err.span_label(ident.span,
                               format!("{} `{}` does not have {} field{}",
                                       kind_name,
                                       tcx.def_path_str(variant.def_id),
                                       t,
                                       plural));
                if plural == "" {
                    let input = unmentioned_fields.iter().map(|field| &field.name);
                    let suggested_name =
                        find_best_match_for_name(input, &ident.as_str(), None);
                    if let Some(suggested_name) = suggested_name {
                        err.span_suggestion(
                            ident.span,
                            "a field with a similar name exists",
                            suggested_name.to_string(),
                            Applicability::MaybeIncorrect,
                        );

                        // we don't want to throw `E0027` in case we have thrown `E0026` for them
                        unmentioned_fields.retain(|&x| x.as_str() != suggested_name.as_str());
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
                     it explicitly."
                );
            }
            err.emit();
        }

        // Require `..` if struct has non_exhaustive attribute.
        if variant.is_field_list_non_exhaustive() && !adt.did.is_local() && !etc {
            span_err!(tcx.sess, span, E0638,
                      "`..` required with {} marked as non-exhaustive",
                      kind_name);
        }

        // Report an error if incorrect number of the fields were specified.
        if kind_name == "union" {
            if fields.len() != 1 {
                tcx.sess.span_err(span, "union patterns should have exactly one field");
            }
            if etc {
                tcx.sess.span_err(span, "`..` cannot be used in union patterns");
            }
        } else if !etc {
            if unmentioned_fields.len() > 0 {
                let field_names = if unmentioned_fields.len() == 1 {
                    format!("field `{}`", unmentioned_fields[0])
                } else {
                    format!("fields {}",
                            unmentioned_fields.iter()
                                .map(|name| format!("`{}`", name))
                                .collect::<Vec<String>>()
                                .join(", "))
                };
                let mut diag = struct_span_err!(tcx.sess, span, E0027,
                                                "pattern does not mention {}",
                                                field_names);
                diag.span_label(span, format!("missing {}", field_names));
                if variant.ctor_kind == CtorKind::Fn {
                    diag.note("trying to match a tuple variant with a struct variant pattern");
                }
                if tcx.sess.teach(&diag.get_code().unwrap()) {
                    diag.note(
                        "This error indicates that a pattern for a struct fails to specify a \
                         sub-pattern for every one of the struct's fields. Ensure that each field \
                         from the struct's definition is mentioned in the pattern, or use `..` to \
                         ignore unwanted fields."
                    );
                }
                diag.emit();
            }
        }
        no_field_errors
    }
}
