use std::iter;

use rustc_abi::{BackendRepr, TagEncoding, Variants, WrappingRange};
use rustc_hir::{Expr, ExprKind, HirId, LangItem};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, SizeSkeleton};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_session::{declare_lint, declare_lint_pass, impl_lint_pass};
use rustc_span::{Span, Symbol, sym};
use tracing::debug;
use {rustc_ast as ast, rustc_hir as hir};

mod improper_ctypes; // these filed do the implementation for ImproperCTypesDefinitions,ImproperCTypesDeclarations
pub(crate) use improper_ctypes::ImproperCTypesLint;

use crate::lints::{
    AmbiguousWidePointerComparisons, AmbiguousWidePointerComparisonsAddrMetadataSuggestion,
    AmbiguousWidePointerComparisonsAddrSuggestion, AmbiguousWidePointerComparisonsCastSuggestion,
    AmbiguousWidePointerComparisonsExpectSuggestion, AtomicOrderingFence, AtomicOrderingLoad,
    AtomicOrderingStore, InvalidAtomicOrderingDiag, InvalidNanComparisons,
    InvalidNanComparisonsSuggestion, UnpredictableFunctionPointerComparisons,
    UnpredictableFunctionPointerComparisonsSuggestion, UnusedComparisons,
    VariantSizeDifferencesDiag,
};
use crate::{LateContext, LateLintPass, LintContext};

mod literal;

use literal::{int_ty_range, lint_literal, uint_ty_range};

declare_lint! {
    /// The `unused_comparisons` lint detects comparisons made useless by
    /// limits of the types involved.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(x: u8) {
    ///     x >= 0;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A useless comparison may indicate a mistake, and should be fixed or
    /// removed.
    UNUSED_COMPARISONS,
    Warn,
    "comparisons made useless by limits of the types involved"
}

declare_lint! {
    /// The `overflowing_literals` lint detects literals out of range for their type.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// let x: u8 = 1000;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to use a literal that overflows its type
    /// Change either the literal or its type such that the literal is
    /// within the range of its type.
    OVERFLOWING_LITERALS,
    Deny,
    "literal out of range for its type"
}

declare_lint! {
    /// The `variant_size_differences` lint detects enums with widely varying
    /// variant sizes.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(variant_size_differences)]
    /// enum En {
    ///     V0(u8),
    ///     VBig([u8; 1024]),
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It can be a mistake to add a variant to an enum that is much larger
    /// than the other variants, bloating the overall size required for all
    /// variants. This can impact performance and memory usage. This is
    /// triggered if one variant is more than 3 times larger than the
    /// second-largest variant.
    ///
    /// Consider placing the large variant's contents on the heap (for example
    /// via [`Box`]) to keep the overall size of the enum itself down.
    ///
    /// This lint is "allow" by default because it can be noisy, and may not be
    /// an actual problem. Decisions about this should be guided with
    /// profiling and benchmarking.
    ///
    /// [`Box`]: https://doc.rust-lang.org/std/boxed/index.html
    VARIANT_SIZE_DIFFERENCES,
    Allow,
    "detects enums with widely varying variant sizes"
}

declare_lint! {
    /// The `invalid_nan_comparisons` lint checks comparison with `f32::NAN` or `f64::NAN`
    /// as one of the operand.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let a = 2.3f32;
    /// if a == f32::NAN {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// NaN does not compare meaningfully to anything – not
    /// even itself – so those comparisons are always false.
    INVALID_NAN_COMPARISONS,
    Warn,
    "detects invalid floating point NaN comparisons"
}

declare_lint! {
    /// The `ambiguous_wide_pointer_comparisons` lint checks comparison
    /// of `*const/*mut ?Sized` as the operands.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # struct A;
    /// # struct B;
    ///
    /// # trait T {}
    /// # impl T for A {}
    /// # impl T for B {}
    ///
    /// let ab = (A, B);
    /// let a = &ab.0 as *const dyn T;
    /// let b = &ab.1 as *const dyn T;
    ///
    /// let _ = a == b;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The comparison includes metadata which may not be expected.
    AMBIGUOUS_WIDE_POINTER_COMPARISONS,
    Warn,
    "detects ambiguous wide pointer comparisons"
}

declare_lint! {
    /// The `unpredictable_function_pointer_comparisons` lint checks comparison
    /// of function pointer as the operands.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn a() {}
    /// fn b() {}
    ///
    /// let f: fn() = a;
    /// let g: fn() = b;
    ///
    /// let _ = f == g;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Function pointers comparisons do not produce meaningful result since
    /// they are never guaranteed to be unique and could vary between different
    /// code generation units. Furthermore, different functions could have the
    /// same address after being merged together.
    UNPREDICTABLE_FUNCTION_POINTER_COMPARISONS,
    Warn,
    "detects unpredictable function pointer comparisons",
    report_in_external_macro
}

#[derive(Copy, Clone, Default)]
pub(crate) struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: Option<hir::HirId>,
    /// Span of the last visited negated expression
    negated_expr_span: Option<Span>,
}

impl_lint_pass!(TypeLimits => [
    UNUSED_COMPARISONS,
    OVERFLOWING_LITERALS,
    INVALID_NAN_COMPARISONS,
    AMBIGUOUS_WIDE_POINTER_COMPARISONS,
    UNPREDICTABLE_FUNCTION_POINTER_COMPARISONS
]);

impl TypeLimits {
    pub(crate) fn new() -> TypeLimits {
        TypeLimits { negated_expr_id: None, negated_expr_span: None }
    }
}

fn lint_nan<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    binop: hir::BinOpKind,
    l: &'tcx hir::Expr<'tcx>,
    r: &'tcx hir::Expr<'tcx>,
) {
    fn is_nan(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
        let expr = expr.peel_blocks().peel_borrows();
        match expr.kind {
            ExprKind::Path(qpath) => {
                let Some(def_id) = cx.typeck_results().qpath_res(&qpath, expr.hir_id).opt_def_id()
                else {
                    return false;
                };

                matches!(
                    cx.tcx.get_diagnostic_name(def_id),
                    Some(sym::f16_nan | sym::f32_nan | sym::f64_nan | sym::f128_nan)
                )
            }
            _ => false,
        }
    }

    fn eq_ne(
        e: &hir::Expr<'_>,
        l: &hir::Expr<'_>,
        r: &hir::Expr<'_>,
        f: impl FnOnce(Span, Span) -> InvalidNanComparisonsSuggestion,
    ) -> InvalidNanComparisons {
        let suggestion = if let Some(l_span) = l.span.find_ancestor_inside(e.span)
            && let Some(r_span) = r.span.find_ancestor_inside(e.span)
        {
            f(l_span, r_span)
        } else {
            InvalidNanComparisonsSuggestion::Spanless
        };

        InvalidNanComparisons::EqNe { suggestion }
    }

    let lint = match binop {
        hir::BinOpKind::Eq | hir::BinOpKind::Ne if is_nan(cx, l) => {
            eq_ne(e, l, r, |l_span, r_span| InvalidNanComparisonsSuggestion::Spanful {
                nan_plus_binop: l_span.until(r_span),
                float: r_span.shrink_to_hi(),
                neg: (binop == hir::BinOpKind::Ne).then(|| r_span.shrink_to_lo()),
            })
        }
        hir::BinOpKind::Eq | hir::BinOpKind::Ne if is_nan(cx, r) => {
            eq_ne(e, l, r, |l_span, r_span| InvalidNanComparisonsSuggestion::Spanful {
                nan_plus_binop: l_span.shrink_to_hi().to(r_span),
                float: l_span.shrink_to_hi(),
                neg: (binop == hir::BinOpKind::Ne).then(|| l_span.shrink_to_lo()),
            })
        }
        hir::BinOpKind::Lt | hir::BinOpKind::Le | hir::BinOpKind::Gt | hir::BinOpKind::Ge
            if is_nan(cx, l) || is_nan(cx, r) =>
        {
            InvalidNanComparisons::LtLeGtGe
        }
        _ => return,
    };

    cx.emit_span_lint(INVALID_NAN_COMPARISONS, e.span, lint);
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum ComparisonOp {
    BinOp(hir::BinOpKind),
    Other,
}

fn lint_wide_pointer<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    cmpop: ComparisonOp,
    l: &'tcx hir::Expr<'tcx>,
    r: &'tcx hir::Expr<'tcx>,
) {
    let ptr_unsized = |mut ty: Ty<'tcx>| -> Option<(
        /* number of refs */ usize,
        /* modifiers */ String,
        /* is dyn */ bool,
    )> {
        let mut refs = 0;
        // here we remove any "implicit" references and count the number
        // of them to correctly suggest the right number of deref
        while let ty::Ref(_, inner_ty, _) = ty.kind() {
            ty = *inner_ty;
            refs += 1;
        }

        // get the inner type of a pointer (or akin)
        let mut modifiers = String::new();
        ty = match ty.kind() {
            ty::RawPtr(ty, _) => *ty,
            ty::Adt(def, args) if cx.tcx.is_diagnostic_item(sym::NonNull, def.did()) => {
                modifiers.push_str(".as_ptr()");
                args.type_at(0)
            }
            _ => return None,
        };

        (!ty.is_sized(cx.tcx, cx.typing_env()))
            .then(|| (refs, modifiers, matches!(ty.kind(), ty::Dynamic(_, _))))
    };

    // the left and right operands can have references, remove any explicit references
    let l = l.peel_borrows();
    let r = r.peel_borrows();

    let Some(l_ty) = cx.typeck_results().expr_ty_opt(l) else {
        return;
    };
    let Some(r_ty) = cx.typeck_results().expr_ty_opt(r) else {
        return;
    };

    let Some((l_ty_refs, l_modifiers, l_inner_ty_is_dyn)) = ptr_unsized(l_ty) else {
        return;
    };
    let Some((r_ty_refs, r_modifiers, r_inner_ty_is_dyn)) = ptr_unsized(r_ty) else {
        return;
    };

    let (Some(l_span), Some(r_span)) =
        (l.span.find_ancestor_inside(e.span), r.span.find_ancestor_inside(e.span))
    else {
        return cx.emit_span_lint(
            AMBIGUOUS_WIDE_POINTER_COMPARISONS,
            e.span,
            AmbiguousWidePointerComparisons::Spanless,
        );
    };

    let ne = if cmpop == ComparisonOp::BinOp(hir::BinOpKind::Ne) { "!" } else { "" };
    let is_eq_ne = matches!(cmpop, ComparisonOp::BinOp(hir::BinOpKind::Eq | hir::BinOpKind::Ne));
    let is_dyn_comparison = l_inner_ty_is_dyn && r_inner_ty_is_dyn;
    let via_method_call = matches!(&e.kind, ExprKind::MethodCall(..) | ExprKind::Call(..));

    let left = e.span.shrink_to_lo().until(l_span.shrink_to_lo());
    let middle = l_span.shrink_to_hi().until(r_span.shrink_to_lo());
    let right = r_span.shrink_to_hi().until(e.span.shrink_to_hi());

    let deref_left = &*"*".repeat(l_ty_refs);
    let deref_right = &*"*".repeat(r_ty_refs);

    let l_modifiers = &*l_modifiers;
    let r_modifiers = &*r_modifiers;

    cx.emit_span_lint(
        AMBIGUOUS_WIDE_POINTER_COMPARISONS,
        e.span,
        if is_eq_ne {
            AmbiguousWidePointerComparisons::SpanfulEq {
                addr_metadata_suggestion: (!is_dyn_comparison).then(|| {
                    AmbiguousWidePointerComparisonsAddrMetadataSuggestion {
                        ne,
                        deref_left,
                        deref_right,
                        l_modifiers,
                        r_modifiers,
                        left,
                        middle,
                        right,
                    }
                }),
                addr_suggestion: AmbiguousWidePointerComparisonsAddrSuggestion {
                    ne,
                    deref_left,
                    deref_right,
                    l_modifiers,
                    r_modifiers,
                    left,
                    middle,
                    right,
                },
            }
        } else {
            AmbiguousWidePointerComparisons::SpanfulCmp {
                cast_suggestion: AmbiguousWidePointerComparisonsCastSuggestion {
                    deref_left,
                    deref_right,
                    l_modifiers,
                    r_modifiers,
                    paren_left: if l_ty_refs != 0 { ")" } else { "" },
                    paren_right: if r_ty_refs != 0 { ")" } else { "" },
                    left_before: (l_ty_refs != 0).then_some(l_span.shrink_to_lo()),
                    left_after: l_span.shrink_to_hi(),
                    right_before: (r_ty_refs != 0).then_some(r_span.shrink_to_lo()),
                    right_after: r_span.shrink_to_hi(),
                },
                expect_suggestion: AmbiguousWidePointerComparisonsExpectSuggestion {
                    paren_left: if via_method_call { "" } else { "(" },
                    paren_right: if via_method_call { "" } else { ")" },
                    before: e.span.shrink_to_lo(),
                    after: e.span.shrink_to_hi(),
                },
            }
        },
    );
}

fn lint_fn_pointer<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    cmpop: ComparisonOp,
    l: &'tcx hir::Expr<'tcx>,
    r: &'tcx hir::Expr<'tcx>,
) {
    let peel_refs = |mut ty: Ty<'tcx>| -> (Ty<'tcx>, usize) {
        let mut refs = 0;

        while let ty::Ref(_, inner_ty, _) = ty.kind() {
            ty = *inner_ty;
            refs += 1;
        }

        (ty, refs)
    };

    // Left and right operands can have borrows, remove them
    let l = l.peel_borrows();
    let r = r.peel_borrows();

    let Some(l_ty) = cx.typeck_results().expr_ty_opt(l) else { return };
    let Some(r_ty) = cx.typeck_results().expr_ty_opt(r) else { return };

    // Remove any references as `==` will deref through them (and count the
    // number of references removed, for latter).
    let (l_ty, l_ty_refs) = peel_refs(l_ty);
    let (r_ty, r_ty_refs) = peel_refs(r_ty);

    if l_ty.is_fn() && r_ty.is_fn() {
        // both operands are function pointers, fallthrough
    } else if let ty::Adt(l_def, l_args) = l_ty.kind()
        && let ty::Adt(r_def, r_args) = r_ty.kind()
        && cx.tcx.is_lang_item(l_def.did(), LangItem::Option)
        && cx.tcx.is_lang_item(r_def.did(), LangItem::Option)
        && let Some(l_some_arg) = l_args.get(0)
        && let Some(r_some_arg) = r_args.get(0)
        && l_some_arg.expect_ty().is_fn()
        && r_some_arg.expect_ty().is_fn()
    {
        // both operands are `Option<{function ptr}>`
        return cx.emit_span_lint(
            UNPREDICTABLE_FUNCTION_POINTER_COMPARISONS,
            e.span,
            UnpredictableFunctionPointerComparisons::Warn,
        );
    } else {
        // types are not function pointers, nothing to do
        return;
    }

    // Let's try to suggest `ptr::fn_addr_eq` if/when possible.

    let is_eq_ne = matches!(cmpop, ComparisonOp::BinOp(hir::BinOpKind::Eq | hir::BinOpKind::Ne));

    if !is_eq_ne {
        // Neither `==` nor `!=`, we can't suggest `ptr::fn_addr_eq`, just show the warning.
        return cx.emit_span_lint(
            UNPREDICTABLE_FUNCTION_POINTER_COMPARISONS,
            e.span,
            UnpredictableFunctionPointerComparisons::Warn,
        );
    }

    let (Some(l_span), Some(r_span)) =
        (l.span.find_ancestor_inside(e.span), r.span.find_ancestor_inside(e.span))
    else {
        // No appropriate spans for the left and right operands, just show the warning.
        return cx.emit_span_lint(
            UNPREDICTABLE_FUNCTION_POINTER_COMPARISONS,
            e.span,
            UnpredictableFunctionPointerComparisons::Warn,
        );
    };

    let ne = if cmpop == ComparisonOp::BinOp(hir::BinOpKind::Ne) { "!" } else { "" };

    // `ptr::fn_addr_eq` only works with raw pointer, deref any references.
    let deref_left = &*"*".repeat(l_ty_refs);
    let deref_right = &*"*".repeat(r_ty_refs);

    let left = e.span.shrink_to_lo().until(l_span.shrink_to_lo());
    let middle = l_span.shrink_to_hi().until(r_span.shrink_to_lo());
    let right = r_span.shrink_to_hi().until(e.span.shrink_to_hi());

    let sugg =
        // We only check for a right cast as `FnDef` == `FnPtr` is not possible,
        // only `FnPtr == FnDef` is possible.
        if !r_ty.is_fn_ptr() {
            let fn_sig = r_ty.fn_sig(cx.tcx);

            UnpredictableFunctionPointerComparisonsSuggestion::FnAddrEqWithCast {
                ne,
                fn_sig,
                deref_left,
                deref_right,
                left,
                middle,
                right,
            }
        } else {
            UnpredictableFunctionPointerComparisonsSuggestion::FnAddrEq {
                ne,
                deref_left,
                deref_right,
                left,
                middle,
                right,
            }
        };

    cx.emit_span_lint(
        UNPREDICTABLE_FUNCTION_POINTER_COMPARISONS,
        e.span,
        UnpredictableFunctionPointerComparisons::Suggestion { sugg },
    );
}

impl<'tcx> LateLintPass<'tcx> for TypeLimits {
    fn check_lit(&mut self, cx: &LateContext<'tcx>, hir_id: HirId, lit: hir::Lit, negated: bool) {
        if negated {
            self.negated_expr_id = Some(hir_id);
            self.negated_expr_span = Some(lit.span);
        }
        lint_literal(cx, self, hir_id, lit.span, &lit, negated);
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx hir::Expr<'tcx>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::Neg, expr) => {
                // Propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != Some(e.hir_id) {
                    self.negated_expr_id = Some(expr.hir_id);
                    self.negated_expr_span = Some(e.span);
                }
            }
            hir::ExprKind::Binary(binop, ref l, ref r) => {
                if is_comparison(binop.node) {
                    if !check_limits(cx, binop.node, l, r) {
                        cx.emit_span_lint(UNUSED_COMPARISONS, e.span, UnusedComparisons);
                    } else {
                        lint_nan(cx, e, binop.node, l, r);
                        let cmpop = ComparisonOp::BinOp(binop.node);
                        lint_wide_pointer(cx, e, cmpop, l, r);
                        lint_fn_pointer(cx, e, cmpop, l, r);
                    }
                }
            }
            hir::ExprKind::Call(path, [l, r])
                if let ExprKind::Path(ref qpath) = path.kind
                    && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                    && let Some(diag_item) = cx.tcx.get_diagnostic_name(def_id)
                    && let Some(cmpop) = diag_item_cmpop(diag_item) =>
            {
                lint_wide_pointer(cx, e, cmpop, l, r);
                lint_fn_pointer(cx, e, cmpop, l, r);
            }
            hir::ExprKind::MethodCall(_, l, [r], _)
                if let Some(def_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
                    && let Some(diag_item) = cx.tcx.get_diagnostic_name(def_id)
                    && let Some(cmpop) = diag_item_cmpop(diag_item) =>
            {
                lint_wide_pointer(cx, e, cmpop, l, r);
                lint_fn_pointer(cx, e, cmpop, l, r);
            }
            _ => {}
        };

        fn is_valid<T: PartialOrd>(binop: hir::BinOpKind, v: T, min: T, max: T) -> bool {
            match binop {
                hir::BinOpKind::Lt => v > min && v <= max,
                hir::BinOpKind::Le => v >= min && v < max,
                hir::BinOpKind::Gt => v >= min && v < max,
                hir::BinOpKind::Ge => v > min && v <= max,
                hir::BinOpKind::Eq | hir::BinOpKind::Ne => v >= min && v <= max,
                _ => bug!(),
            }
        }

        fn rev_binop(binop: hir::BinOpKind) -> hir::BinOpKind {
            match binop {
                hir::BinOpKind::Lt => hir::BinOpKind::Gt,
                hir::BinOpKind::Le => hir::BinOpKind::Ge,
                hir::BinOpKind::Gt => hir::BinOpKind::Lt,
                hir::BinOpKind::Ge => hir::BinOpKind::Le,
                _ => binop,
            }
        }

        fn check_limits(
            cx: &LateContext<'_>,
            binop: hir::BinOpKind,
            l: &hir::Expr<'_>,
            r: &hir::Expr<'_>,
        ) -> bool {
            let (lit, expr, swap) = match (&l.kind, &r.kind) {
                (&hir::ExprKind::Lit(_), _) => (l, r, true),
                (_, &hir::ExprKind::Lit(_)) => (r, l, false),
                _ => return true,
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap { rev_binop(binop) } else { binop };
            match *cx.typeck_results().node_type(expr.hir_id).kind() {
                ty::Int(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i128 = match lit.kind {
                        hir::ExprKind::Lit(li) => match li.node {
                            ast::LitKind::Int(
                                v,
                                ast::LitIntType::Signed(_) | ast::LitIntType::Unsuffixed,
                            ) => v.get() as i128,
                            _ => return true,
                        },
                        _ => bug!(),
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                ty::Uint(uint_ty) => {
                    let (min, max): (u128, u128) = uint_ty_range(uint_ty);
                    let lit_val: u128 = match lit.kind {
                        hir::ExprKind::Lit(li) => match li.node {
                            ast::LitKind::Int(v, _) => v.get(),
                            _ => return true,
                        },
                        _ => bug!(),
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                _ => true,
            }
        }

        fn is_comparison(binop: hir::BinOpKind) -> bool {
            matches!(
                binop,
                hir::BinOpKind::Eq
                    | hir::BinOpKind::Lt
                    | hir::BinOpKind::Le
                    | hir::BinOpKind::Ne
                    | hir::BinOpKind::Ge
                    | hir::BinOpKind::Gt
            )
        }

        fn diag_item_cmpop(diag_item: Symbol) -> Option<ComparisonOp> {
            Some(match diag_item {
                sym::cmp_ord_max => ComparisonOp::Other,
                sym::cmp_ord_min => ComparisonOp::Other,
                sym::ord_cmp_method => ComparisonOp::Other,
                sym::cmp_partialeq_eq => ComparisonOp::BinOp(hir::BinOpKind::Eq),
                sym::cmp_partialeq_ne => ComparisonOp::BinOp(hir::BinOpKind::Ne),
                sym::cmp_partialord_cmp => ComparisonOp::Other,
                sym::cmp_partialord_ge => ComparisonOp::BinOp(hir::BinOpKind::Ge),
                sym::cmp_partialord_gt => ComparisonOp::BinOp(hir::BinOpKind::Gt),
                sym::cmp_partialord_le => ComparisonOp::BinOp(hir::BinOpKind::Le),
                sym::cmp_partialord_lt => ComparisonOp::BinOp(hir::BinOpKind::Lt),
                _ => return None,
            })
        }
    }
}

pub(crate) fn nonnull_optimization_guaranteed<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::AdtDef<'tcx>,
) -> bool {
    tcx.has_attr(def.did(), sym::rustc_nonnull_optimization_guaranteed)
}

/// `repr(transparent)` structs can have a single non-1-ZST field, this function returns that
/// field.
pub(crate) fn transparent_newtype_field<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    variant: &'a ty::VariantDef,
) -> Option<&'a ty::FieldDef> {
    let typing_env = ty::TypingEnv::non_body_analysis(tcx, variant.def_id);
    variant.fields.iter().find(|field| {
        let field_ty = tcx.type_of(field.did).instantiate_identity();
        let is_1zst =
            tcx.layout_of(typing_env.as_query_input(field_ty)).is_ok_and(|layout| layout.is_1zst());
        !is_1zst
    })
}

/// Is type known to be non-null?
fn ty_is_known_nonnull<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);

    match ty.kind() {
        ty::FnPtr(..) => true,
        ty::Ref(..) => true,
        ty::Adt(def, _) if def.is_box() => true,
        ty::Adt(def, args) if def.repr().transparent() && !def.is_union() => {
            let marked_non_null = nonnull_optimization_guaranteed(tcx, *def);

            if marked_non_null {
                return true;
            }

            // `UnsafeCell` and `UnsafePinned` have their niche hidden.
            if def.is_unsafe_cell() || def.is_unsafe_pinned() {
                return false;
            }

            def.variants()
                .iter()
                .filter_map(|variant| transparent_newtype_field(tcx, variant))
                .any(|field| ty_is_known_nonnull(tcx, typing_env, field.ty(tcx, args)))
        }
        ty::Pat(base, pat) => {
            ty_is_known_nonnull(tcx, typing_env, *base)
                || pat_ty_is_known_nonnull(tcx, typing_env, *pat)
        }
        _ => false,
    }
}

fn pat_ty_is_known_nonnull<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    pat: ty::Pattern<'tcx>,
) -> bool {
    Option::unwrap_or_default(
        try {
            match *pat {
                ty::PatternKind::Range { start, end } => {
                    let start = start.try_to_value()?.try_to_bits(tcx, typing_env)?;
                    let end = end.try_to_value()?.try_to_bits(tcx, typing_env)?;

                    // This also works for negative numbers, as we just need
                    // to ensure we aren't wrapping over zero.
                    start > 0 && end >= start
                }
                ty::PatternKind::Or(patterns) => {
                    patterns.iter().all(|pat| pat_ty_is_known_nonnull(tcx, typing_env, pat))
                }
            }
        },
    )
}

/// Given a non-null scalar (or transparent) type `ty`, return the nullable version of that type.
/// If the type passed in was not scalar, returns None.
fn get_nullable_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<Ty<'tcx>> {
    let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);

    Some(match *ty.kind() {
        ty::Adt(field_def, field_args) => {
            let inner_field_ty = {
                let mut first_non_zst_ty =
                    field_def.variants().iter().filter_map(|v| transparent_newtype_field(tcx, v));
                debug_assert_eq!(
                    first_non_zst_ty.clone().count(),
                    1,
                    "Wrong number of fields for transparent type"
                );
                first_non_zst_ty
                    .next_back()
                    .expect("No non-zst fields in transparent type.")
                    .ty(tcx, field_args)
            };
            return get_nullable_type(tcx, typing_env, inner_field_ty);
        }
        ty::Pat(base, ..) => return get_nullable_type(tcx, typing_env, base),
        ty::Int(_) | ty::Uint(_) | ty::RawPtr(..) => ty,
        // As these types are always non-null, the nullable equivalent of
        // `Option<T>` of these types are their raw pointer counterparts.
        ty::Ref(_region, ty, mutbl) => Ty::new_ptr(tcx, ty, mutbl),
        // There is no nullable equivalent for Rust's function pointers,
        // you must use an `Option<fn(..) -> _>` to represent it.
        ty::FnPtr(..) => ty,
        // We should only ever reach this case if `ty_is_known_nonnull` is
        // extended to other types.
        ref unhandled => {
            debug!(
                "get_nullable_type: Unhandled scalar kind: {:?} while checking {:?}",
                unhandled, ty
            );
            return None;
        }
    })
}

/// A type is niche-optimization candidate iff:
/// - Is a zero-sized type with alignment 1 (a “1-ZST”).
/// - Has no fields.
/// - Does not have the `#[non_exhaustive]` attribute.
fn is_niche_optimization_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    if tcx.layout_of(typing_env.as_query_input(ty)).is_ok_and(|layout| !layout.is_1zst()) {
        return false;
    }

    match ty.kind() {
        ty::Adt(ty_def, _) => {
            let non_exhaustive = ty_def.is_variant_list_non_exhaustive();
            let empty = (ty_def.is_struct() && ty_def.all_fields().next().is_none())
                || (ty_def.is_enum() && ty_def.variants().is_empty());

            !non_exhaustive && empty
        }
        ty::Tuple(tys) => tys.is_empty(),
        _ => false,
    }
}

/// Check if this enum can be safely exported based on the "nullable pointer optimization". If it
/// can, return the type that `ty` can be safely converted to, otherwise return `None`.
/// Currently restricted to function pointers, boxes, references, `core::num::NonZero`,
/// `core::ptr::NonNull`, and `#[repr(transparent)]` newtypes.
pub(crate) fn repr_nullable_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<Ty<'tcx>> {
    debug!("is_repr_nullable_ptr(tcx, ty = {:?})", ty);
    match ty.kind() {
        ty::Adt(ty_def, args) => {
            let field_ty = match &ty_def.variants().raw[..] {
                [var_one, var_two] => match (&var_one.fields.raw[..], &var_two.fields.raw[..]) {
                    ([], [field]) | ([field], []) => field.ty(tcx, args),
                    ([field1], [field2]) => {
                        let ty1 = field1.ty(tcx, args);
                        let ty2 = field2.ty(tcx, args);

                        if is_niche_optimization_candidate(tcx, typing_env, ty1) {
                            ty2
                        } else if is_niche_optimization_candidate(tcx, typing_env, ty2) {
                            ty1
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                },
                _ => return None,
            };

            if !ty_is_known_nonnull(tcx, typing_env, field_ty) {
                return None;
            }

            // At this point, the field's type is known to be nonnull and the parent enum is Option-like.
            // If the computed size for the field and the enum are different, the nonnull optimization isn't
            // being applied (and we've got a problem somewhere).
            let compute_size_skeleton = |t| SizeSkeleton::compute(t, tcx, typing_env).ok();
            if !compute_size_skeleton(ty)?.same_size(compute_size_skeleton(field_ty)?) {
                bug!("improper_ctypes: Option nonnull optimization not applied?");
            }

            // Return the nullable type this Option-like enum can be safely represented with.
            let field_ty_layout = tcx.layout_of(typing_env.as_query_input(field_ty));
            if field_ty_layout.is_err() && !field_ty.has_non_region_param() {
                bug!("should be able to compute the layout of non-polymorphic type");
            }

            let field_ty_abi = &field_ty_layout.ok()?.backend_repr;
            if let BackendRepr::Scalar(field_ty_scalar) = field_ty_abi {
                match field_ty_scalar.valid_range(&tcx) {
                    WrappingRange { start: 0, end }
                        if end == field_ty_scalar.size(&tcx).unsigned_int_max() - 1 =>
                    {
                        return Some(get_nullable_type(tcx, typing_env, field_ty).unwrap());
                    }
                    WrappingRange { start: 1, .. } => {
                        return Some(get_nullable_type(tcx, typing_env, field_ty).unwrap());
                    }
                    WrappingRange { start, end } => {
                        unreachable!("Unhandled start and end range: ({}, {})", start, end)
                    }
                };
            }
            None
        }
        ty::Pat(base, pat) => get_nullable_type_from_pat(tcx, typing_env, *base, *pat),
        _ => None,
    }
}

fn get_nullable_type_from_pat<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    base: Ty<'tcx>,
    pat: ty::Pattern<'tcx>,
) -> Option<Ty<'tcx>> {
    match *pat {
        ty::PatternKind::Range { .. } => get_nullable_type(tcx, typing_env, base),
        ty::PatternKind::Or(patterns) => {
            let first = get_nullable_type_from_pat(tcx, typing_env, base, patterns[0])?;
            for &pat in &patterns[1..] {
                assert_eq!(first, get_nullable_type_from_pat(tcx, typing_env, base, pat)?);
            }
            Some(first)
        }
    }
}

declare_lint_pass!(VariantSizeDifferences => [VARIANT_SIZE_DIFFERENCES]);

impl<'tcx> LateLintPass<'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Enum(_, _, ref enum_definition) = it.kind {
            let t = cx.tcx.type_of(it.owner_id).instantiate_identity();
            let ty = cx.tcx.erase_and_anonymize_regions(t);
            let Ok(layout) = cx.layout_of(ty) else { return };
            let Variants::Multiple { tag_encoding: TagEncoding::Direct, tag, variants, .. } =
                &layout.variants
            else {
                return;
            };

            let tag_size = tag.size(&cx.tcx).bytes();

            debug!(
                "enum `{}` is {} bytes large with layout:\n{:#?}",
                t,
                layout.size.bytes(),
                layout
            );

            let (largest, slargest, largest_index) = iter::zip(enum_definition.variants, variants)
                .map(|(variant, variant_layout)| {
                    // Subtract the size of the enum tag.
                    let bytes = variant_layout.size.bytes().saturating_sub(tag_size);

                    debug!("- variant `{}` is {} bytes large", variant.ident, bytes);
                    bytes
                })
                .enumerate()
                .fold((0, 0, 0), |(l, s, li), (idx, size)| {
                    if size > l {
                        (size, l, idx)
                    } else if size > s {
                        (l, size, li)
                    } else {
                        (l, s, li)
                    }
                });

            // We only warn if the largest variant is at least thrice as large as
            // the second-largest.
            if largest > slargest * 3 && slargest > 0 {
                cx.emit_span_lint(
                    VARIANT_SIZE_DIFFERENCES,
                    enum_definition.variants[largest_index].span,
                    VariantSizeDifferencesDiag { largest },
                );
            }
        }
    }
}

declare_lint! {
    /// The `invalid_atomic_ordering` lint detects passing an `Ordering`
    /// to an atomic operation that does not support that ordering.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # use core::sync::atomic::{AtomicU8, Ordering};
    /// let atom = AtomicU8::new(0);
    /// let value = atom.load(Ordering::Release);
    /// # let _ = value;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Some atomic operations are only supported for a subset of the
    /// `atomic::Ordering` variants. Passing an unsupported variant will cause
    /// an unconditional panic at runtime, which is detected by this lint.
    ///
    /// This lint will trigger in the following cases: (where `AtomicType` is an
    /// atomic type from `core::sync::atomic`, such as `AtomicBool`,
    /// `AtomicPtr`, `AtomicUsize`, or any of the other integer atomics).
    ///
    /// - Passing `Ordering::Acquire` or `Ordering::AcqRel` to
    ///   `AtomicType::store`.
    ///
    /// - Passing `Ordering::Release` or `Ordering::AcqRel` to
    ///   `AtomicType::load`.
    ///
    /// - Passing `Ordering::Relaxed` to `core::sync::atomic::fence` or
    ///   `core::sync::atomic::compiler_fence`.
    ///
    /// - Passing `Ordering::Release` or `Ordering::AcqRel` as the failure
    ///   ordering for any of `AtomicType::compare_exchange`,
    ///   `AtomicType::compare_exchange_weak`, or `AtomicType::fetch_update`.
    INVALID_ATOMIC_ORDERING,
    Deny,
    "usage of invalid atomic ordering in atomic operations and memory fences"
}

declare_lint_pass!(InvalidAtomicOrdering => [INVALID_ATOMIC_ORDERING]);

impl InvalidAtomicOrdering {
    fn inherent_atomic_method_call<'hir>(
        cx: &LateContext<'_>,
        expr: &Expr<'hir>,
        recognized_names: &[Symbol], // used for fast path calculation
    ) -> Option<(Symbol, &'hir [Expr<'hir>])> {
        const ATOMIC_TYPES: &[Symbol] = &[
            sym::AtomicBool,
            sym::AtomicPtr,
            sym::AtomicUsize,
            sym::AtomicU8,
            sym::AtomicU16,
            sym::AtomicU32,
            sym::AtomicU64,
            sym::AtomicU128,
            sym::AtomicIsize,
            sym::AtomicI8,
            sym::AtomicI16,
            sym::AtomicI32,
            sym::AtomicI64,
            sym::AtomicI128,
        ];
        if let ExprKind::MethodCall(method_path, _, args, _) = &expr.kind
            && recognized_names.contains(&method_path.ident.name)
            && let Some(m_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
            // skip extension traits, only lint functions from the standard library
            && let Some(impl_did) = cx.tcx.inherent_impl_of_assoc(m_def_id)
            && let Some(adt) = cx.tcx.type_of(impl_did).instantiate_identity().ty_adt_def()
            && let parent = cx.tcx.parent(adt.did())
            && cx.tcx.is_diagnostic_item(sym::atomic_mod, parent)
            && ATOMIC_TYPES.contains(&cx.tcx.item_name(adt.did()))
        {
            return Some((method_path.ident.name, args));
        }
        None
    }

    fn match_ordering(cx: &LateContext<'_>, ord_arg: &Expr<'_>) -> Option<Symbol> {
        let ExprKind::Path(ref ord_qpath) = ord_arg.kind else { return None };
        let did = cx.qpath_res(ord_qpath, ord_arg.hir_id).opt_def_id()?;
        let tcx = cx.tcx;
        let atomic_ordering = tcx.get_diagnostic_item(sym::Ordering);
        let name = tcx.item_name(did);
        let parent = tcx.parent(did);
        [sym::Relaxed, sym::Release, sym::Acquire, sym::AcqRel, sym::SeqCst].into_iter().find(
            |&ordering| {
                name == ordering
                    && (Some(parent) == atomic_ordering
                            // needed in case this is a ctor, not a variant
                            || tcx.opt_parent(parent) == atomic_ordering)
            },
        )
    }

    fn check_atomic_load_store(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let Some((method, args)) =
            Self::inherent_atomic_method_call(cx, expr, &[sym::load, sym::store])
            && let Some((ordering_arg, invalid_ordering)) = match method {
                sym::load => Some((&args[0], sym::Release)),
                sym::store => Some((&args[1], sym::Acquire)),
                _ => None,
            }
            && let Some(ordering) = Self::match_ordering(cx, ordering_arg)
            && (ordering == invalid_ordering || ordering == sym::AcqRel)
        {
            if method == sym::load {
                cx.emit_span_lint(INVALID_ATOMIC_ORDERING, ordering_arg.span, AtomicOrderingLoad);
            } else {
                cx.emit_span_lint(INVALID_ATOMIC_ORDERING, ordering_arg.span, AtomicOrderingStore);
            };
        }
    }

    fn check_memory_fence(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Call(func, args) = expr.kind
            && let ExprKind::Path(ref func_qpath) = func.kind
            && let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id()
            && matches!(cx.tcx.get_diagnostic_name(def_id), Some(sym::fence | sym::compiler_fence))
            && Self::match_ordering(cx, &args[0]) == Some(sym::Relaxed)
        {
            cx.emit_span_lint(INVALID_ATOMIC_ORDERING, args[0].span, AtomicOrderingFence);
        }
    }

    fn check_atomic_compare_exchange(cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some((method, args)) = Self::inherent_atomic_method_call(
            cx,
            expr,
            &[sym::fetch_update, sym::compare_exchange, sym::compare_exchange_weak],
        ) else {
            return;
        };

        let fail_order_arg = match method {
            sym::fetch_update => &args[1],
            sym::compare_exchange | sym::compare_exchange_weak => &args[3],
            _ => return,
        };

        let Some(fail_ordering) = Self::match_ordering(cx, fail_order_arg) else { return };

        if matches!(fail_ordering, sym::Release | sym::AcqRel) {
            cx.emit_span_lint(
                INVALID_ATOMIC_ORDERING,
                fail_order_arg.span,
                InvalidAtomicOrderingDiag { method, fail_order_arg_span: fail_order_arg.span },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for InvalidAtomicOrdering {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        Self::check_atomic_load_store(cx, expr);
        Self::check_memory_fence(cx, expr);
        Self::check_atomic_compare_exchange(cx, expr);
    }
}
