use std::iter;
use std::ops::ControlFlow;

use rustc_abi::{
    BackendRepr, Integer, IntegerType, TagEncoding, VariantIdx, Variants, WrappingRange,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::DiagMessage;
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::{AmbigArg, Expr, ExprKind, HirId, LangItem};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, SizeSkeleton};
use rustc_middle::ty::{
    self, Adt, AdtKind, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt,
};
use rustc_session::{declare_lint, declare_lint_pass, impl_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol, sym};
use tracing::debug;
use {rustc_ast as ast, rustc_hir as hir};

mod improper_ctypes;

use crate::lints::{
    AmbiguousWidePointerComparisons, AmbiguousWidePointerComparisonsAddrMetadataSuggestion,
    AmbiguousWidePointerComparisonsAddrSuggestion, AtomicOrderingFence, AtomicOrderingLoad,
    AtomicOrderingStore, ImproperCTypes, InvalidAtomicOrderingDiag, InvalidNanComparisons,
    InvalidNanComparisonsSuggestion, UnpredictableFunctionPointerComparisons,
    UnpredictableFunctionPointerComparisonsSuggestion, UnusedComparisons, UsesPowerAlignment,
    VariantSizeDifferencesDiag,
};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

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
    /// The `overflowing_literals` lint detects literal out of range for its
    /// type.
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
    /// It is usually a mistake to use a literal that overflows the type where
    /// it is used. Either use a literal that is within range, or change the
    /// type to be within the range of the literal.
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
    "detects unpredictable function pointer comparisons"
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
            .then(|| (refs, modifiers, matches!(ty.kind(), ty::Dynamic(_, _, ty::Dyn))))
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
        AmbiguousWidePointerComparisons::Spanful {
            addr_metadata_suggestion: (is_eq_ne && !is_dyn_comparison).then(|| {
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
            addr_suggestion: if is_eq_ne {
                AmbiguousWidePointerComparisonsAddrSuggestion::AddrEq {
                    ne,
                    deref_left,
                    deref_right,
                    l_modifiers,
                    r_modifiers,
                    left,
                    middle,
                    right,
                }
            } else {
                AmbiguousWidePointerComparisonsAddrSuggestion::Cast {
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
                }
            },
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
    fn check_lit(
        &mut self,
        cx: &LateContext<'tcx>,
        hir_id: HirId,
        lit: &'tcx hir::Lit,
        negated: bool,
    ) {
        if negated {
            self.negated_expr_id = Some(hir_id);
            self.negated_expr_span = Some(lit.span);
        }
        lint_literal(cx, self, hir_id, lit.span, lit, negated);
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

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    ///
    /// ### Example
    ///
    /// ```rust
    /// unsafe extern "C" {
    ///     static STATIC: String;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used in `extern`
    /// blocks are safe and follow certain rules to ensure proper
    /// compatibility with the foreign interfaces. This lint is issued when it
    /// detects a probable mistake in a definition. The lint usually should
    /// provide a description of the issue, along with possibly a hint on how
    /// to resolve it.
    IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

declare_lint_pass!(ImproperCTypesDeclarations => [IMPROPER_CTYPES]);

declare_lint! {
    /// The `improper_ctypes_definitions` lint detects incorrect use of
    /// [`extern` function] definitions.
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub extern "C" fn str_type(p: &str) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    IMPROPER_CTYPES_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign item definitions"
}

declare_lint! {
    /// The `uses_power_alignment` lint detects specific `repr(C)`
    /// aggregates on AIX.
    /// In its platform C ABI, AIX uses the "power" (as in PowerPC) alignment
    /// rule (detailed in https://www.ibm.com/docs/en/xl-c-and-cpp-aix/16.1?topic=data-using-alignment-modes#alignment),
    /// which can also be set for XLC by `#pragma align(power)` or
    /// `-qalign=power`. Aggregates with a floating-point type as the
    /// recursively first field (as in "at offset 0") modify the layout of
    /// *subsequent* fields of the associated structs to use an alignment value
    /// where the floating-point type is aligned on a 4-byte boundary.
    ///
    /// The power alignment rule for structs needed for C compatibility is
    /// unimplementable within `repr(C)` in the compiler without building in
    /// handling of references to packed fields and infectious nested layouts,
    /// so a warning is produced in these situations.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-powerpc64-ibm-aix)
    /// #[repr(C)]
    /// pub struct Floats {
    ///     a: f64,
    ///     b: u8,
    ///     c: f64,
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: repr(C) does not follow the power alignment rule. This may affect platform C ABI compatibility for this type
    ///  --> <source>:5:3
    ///   |
    /// 5 |   c: f64,
    ///   |   ^^^^^^
    ///   |
    ///   = note: `#[warn(uses_power_alignment)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The power alignment rule specifies that the above struct has the
    /// following alignment:
    ///  - offset_of!(Floats, a) == 0
    ///  - offset_of!(Floats, b) == 8
    ///  - offset_of!(Floats, c) == 12
    /// However, rust currently aligns `c` at offset_of!(Floats, c) == 16.
    /// Thus, a warning should be produced for the above struct in this case.
    USES_POWER_ALIGNMENT,
    Warn,
    "Structs do not follow the power alignment rule under repr(C)"
}

declare_lint_pass!(ImproperCTypesDefinitions => [IMPROPER_CTYPES_DEFINITIONS, USES_POWER_ALIGNMENT]);

#[derive(Clone, Copy)]
pub(crate) enum CItemKind {
    Declaration,
    Definition,
}

struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    mode: CItemKind,
}

/// Accumulator for recursive ffi type checking
struct CTypesVisitorState<'tcx> {
    cache: FxHashSet<Ty<'tcx>>,
    /// The original type being checked, before we recursed
    /// to any other types it contains.
    base_ty: Ty<'tcx>,
}

enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe { ty: Ty<'tcx>, reason: DiagMessage, help: Option<DiagMessage> },
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
    mode: CItemKind,
) -> bool {
    let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);

    match ty.kind() {
        ty::FnPtr(..) => true,
        ty::Ref(..) => true,
        ty::Adt(def, _) if def.is_box() && matches!(mode, CItemKind::Definition) => true,
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
                .any(|field| ty_is_known_nonnull(tcx, typing_env, field.ty(tcx, args), mode))
        }
        ty::Pat(base, pat) => {
            ty_is_known_nonnull(tcx, typing_env, *base, mode)
                || Option::unwrap_or_default(
                    try {
                        match **pat {
                            ty::PatternKind::Range { start, end } => {
                                let start = start.try_to_value()?.try_to_bits(tcx, typing_env)?;
                                let end = end.try_to_value()?.try_to_bits(tcx, typing_env)?;

                                // This also works for negative numbers, as we just need
                                // to ensure we aren't wrapping over zero.
                                start > 0 && end >= start
                            }
                        }
                    },
                )
        }
        _ => false,
    }
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
    ckind: CItemKind,
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

            if !ty_is_known_nonnull(tcx, typing_env, field_ty, ckind) {
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
        ty::Pat(base, pat) => match **pat {
            ty::PatternKind::Range { .. } => get_nullable_type(tcx, typing_env, *base),
        },
        _ => None,
    }
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the type is array and emit an unsafe type lint.
    fn check_for_array_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        if let ty::Array(..) = ty.kind() {
            self.emit_ffi_unsafe_type_lint(
                ty,
                sp,
                fluent::lint_improper_ctypes_array_reason,
                Some(fluent::lint_improper_ctypes_array_help),
            );
            true
        } else {
            false
        }
    }

    /// Checks if the given field's type is "ffi-safe".
    fn check_field_type_for_ffi(
        &self,
        acc: &mut CTypesVisitorState<'tcx>,
        field: &ty::FieldDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        let field_ty = field.ty(self.cx.tcx, args);
        let field_ty = self
            .cx
            .tcx
            .try_normalize_erasing_regions(self.cx.typing_env(), field_ty)
            .unwrap_or(field_ty);
        self.check_type_for_ffi(acc, field_ty)
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn check_variant_for_ffi(
        &self,
        acc: &mut CTypesVisitorState<'tcx>,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;
        let transparent_with_all_zst_fields = if def.repr().transparent() {
            if let Some(field) = transparent_newtype_field(self.cx.tcx, variant) {
                // Transparent newtypes have at most one non-ZST field which needs to be checked..
                match self.check_field_type_for_ffi(acc, field, args) {
                    FfiUnsafe { ty, .. } if ty.is_unit() => (),
                    r => return r,
                }

                false
            } else {
                // ..or have only ZST fields, which is FFI-unsafe (unless those fields are all
                // `PhantomData`).
                true
            }
        } else {
            false
        };

        // We can't completely trust `repr(C)` markings, so make sure the fields are actually safe.
        let mut all_phantom = !variant.fields.is_empty();
        for field in &variant.fields {
            all_phantom &= match self.check_field_type_for_ffi(acc, field, args) {
                FfiSafe => false,
                // `()` fields are FFI-safe!
                FfiUnsafe { ty, .. } if ty.is_unit() => false,
                FfiPhantom(..) => true,
                r @ FfiUnsafe { .. } => return r,
            }
        }

        if all_phantom {
            FfiPhantom(ty)
        } else if transparent_with_all_zst_fields {
            FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_struct_zst, help: None }
        } else {
            FfiSafe
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(
        &self,
        acc: &mut CTypesVisitorState<'tcx>,
        ty: Ty<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        let tcx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !acc.cache.insert(ty) {
            return FfiSafe;
        }

        match *ty.kind() {
            ty::Adt(def, args) => {
                if let Some(boxed) = ty.boxed_ty()
                    && matches!(self.mode, CItemKind::Definition)
                {
                    if boxed.is_sized(tcx, self.cx.typing_env()) {
                        return FfiSafe;
                    } else {
                        return FfiUnsafe {
                            ty,
                            reason: fluent::lint_improper_ctypes_box,
                            help: None,
                        };
                    }
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                            && !acc.base_ty.is_mutable_ptr()
                        {
                            return FfiUnsafe {
                                ty,
                                reason: fluent::lint_improper_ctypes_cstr_reason,
                                help: Some(fluent::lint_improper_ctypes_cstr_help),
                            };
                        }

                        if !def.repr().c() && !def.repr().transparent() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_layout_reason
                                } else {
                                    fluent::lint_improper_ctypes_union_layout_reason
                                },
                                help: if def.is_struct() {
                                    Some(fluent::lint_improper_ctypes_struct_layout_help)
                                } else {
                                    Some(fluent::lint_improper_ctypes_union_layout_help)
                                },
                            };
                        }

                        if def.non_enum_variant().field_list_has_applicable_non_exhaustive() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_non_exhaustive
                                } else {
                                    fluent::lint_improper_ctypes_union_non_exhaustive
                                },
                                help: None,
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_fieldless_reason
                                } else {
                                    fluent::lint_improper_ctypes_union_fieldless_reason
                                },
                                help: if def.is_struct() {
                                    Some(fluent::lint_improper_ctypes_struct_fieldless_help)
                                } else {
                                    Some(fluent::lint_improper_ctypes_union_fieldless_help)
                                },
                            };
                        }

                        self.check_variant_for_ffi(acc, ty, def, def.non_enum_variant(), args)
                    }
                    AdtKind::Enum => {
                        if def.variants().is_empty() {
                            // Empty enums are okay... although sort of useless.
                            return FfiSafe;
                        }
                        // Check for a repr() attribute to specify the size of the
                        // discriminant.
                        if !def.repr().c() && !def.repr().transparent() && def.repr().int.is_none()
                        {
                            // Special-case types like `Option<extern fn()>` and `Result<extern fn(), ()>`
                            if let Some(ty) =
                                repr_nullable_ptr(self.cx.tcx, self.cx.typing_env(), ty, self.mode)
                            {
                                return self.check_type_for_ffi(acc, ty);
                            }

                            return FfiUnsafe {
                                ty,
                                reason: fluent::lint_improper_ctypes_enum_repr_reason,
                                help: Some(fluent::lint_improper_ctypes_enum_repr_help),
                            };
                        }

                        if let Some(IntegerType::Fixed(Integer::I128, _)) = def.repr().int {
                            return FfiUnsafe {
                                ty,
                                reason: fluent::lint_improper_ctypes_128bit,
                                help: None,
                            };
                        }

                        use improper_ctypes::check_non_exhaustive_variant;

                        let non_exhaustive = def.variant_list_has_applicable_non_exhaustive();
                        // Check the contained variants.
                        let ret = def.variants().iter().try_for_each(|variant| {
                            check_non_exhaustive_variant(non_exhaustive, variant)
                                .map_break(|reason| FfiUnsafe { ty, reason, help: None })?;

                            match self.check_variant_for_ffi(acc, ty, def, variant, args) {
                                FfiSafe => ControlFlow::Continue(()),
                                r => ControlFlow::Break(r),
                            }
                        });
                        if let ControlFlow::Break(result) = ret {
                            return result;
                        }

                        FfiSafe
                    }
                }
            }

            ty::Char => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_char_reason,
                help: Some(fluent::lint_improper_ctypes_char_help),
            },

            // It's just extra invariants on the type that you need to uphold,
            // but only the base type is relevant for being representable in FFI.
            ty::Pat(base, ..) => self.check_type_for_ffi(acc, base),

            ty::Int(ty::IntTy::I128) | ty::Uint(ty::UintTy::U128) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_128bit, help: None }
            }

            // Primitive types with a stable representation.
            ty::Bool | ty::Int(..) | ty::Uint(..) | ty::Float(..) | ty::Never => FfiSafe,

            ty::Slice(_) => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_slice_reason,
                help: Some(fluent::lint_improper_ctypes_slice_help),
            },

            ty::Dynamic(..) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_dyn, help: None }
            }

            ty::Str => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_str_reason,
                help: Some(fluent::lint_improper_ctypes_str_help),
            },

            ty::Tuple(..) => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_tuple_reason,
                help: Some(fluent::lint_improper_ctypes_tuple_help),
            },

            ty::RawPtr(ty, _) | ty::Ref(_, ty, _)
                if {
                    matches!(self.mode, CItemKind::Definition)
                        && ty.is_sized(self.cx.tcx, self.cx.typing_env())
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty, _)
                if match ty.kind() {
                    ty::Tuple(tuple) => tuple.is_empty(),
                    _ => false,
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty, _) | ty::Ref(_, ty, _) => self.check_type_for_ffi(acc, ty),

            ty::Array(inner_ty, _) => self.check_type_for_ffi(acc, inner_ty),

            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if sig.abi().is_rustic_abi() {
                    return FfiUnsafe {
                        ty,
                        reason: fluent::lint_improper_ctypes_fnptr_reason,
                        help: Some(fluent::lint_improper_ctypes_fnptr_help),
                    };
                }

                let sig = tcx.instantiate_bound_regions_with_erased(sig);
                for arg in sig.inputs() {
                    match self.check_type_for_ffi(acc, *arg) {
                        FfiSafe => {}
                        r => return r,
                    }
                }

                let ret_ty = sig.output();
                if ret_ty.is_unit() {
                    return FfiSafe;
                }

                self.check_type_for_ffi(acc, ret_ty)
            }

            ty::Foreign(..) => FfiSafe,

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach this branch.
            ty::Alias(ty::Opaque, ..) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_opaque, help: None }
            }

            // `extern "C" fn` functions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent, ..)
                if matches!(self.mode, CItemKind::Definition) =>
            {
                FfiSafe
            }

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Param(..)
            | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Placeholder(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &mut self,
        ty: Ty<'tcx>,
        sp: Span,
        note: DiagMessage,
        help: Option<DiagMessage>,
    ) {
        let lint = match self.mode {
            CItemKind::Declaration => IMPROPER_CTYPES,
            CItemKind::Definition => IMPROPER_CTYPES_DEFINITIONS,
        };
        let desc = match self.mode {
            CItemKind::Declaration => "block",
            CItemKind::Definition => "fn",
        };
        let span_note = if let ty::Adt(def, _) = ty.kind()
            && let Some(sp) = self.cx.tcx.hir_span_if_local(def.did())
        {
            Some(sp)
        } else {
            None
        };
        self.cx.emit_span_lint(
            lint,
            sp,
            ImproperCTypes { ty, desc, label: sp, help, note, span_note },
        );
    }

    fn check_for_opaque_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        struct ProhibitOpaqueTypes;
        impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for ProhibitOpaqueTypes {
            type Result = ControlFlow<Ty<'tcx>>;

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if !ty.has_opaque_types() {
                    return ControlFlow::Continue(());
                }

                if let ty::Alias(ty::Opaque, ..) = ty.kind() {
                    ControlFlow::Break(ty)
                } else {
                    ty.super_visit_with(self)
                }
            }
        }

        if let Some(ty) = self
            .cx
            .tcx
            .try_normalize_erasing_regions(self.cx.typing_env(), ty)
            .unwrap_or(ty)
            .visit_with(&mut ProhibitOpaqueTypes)
            .break_value()
        {
            self.emit_ffi_unsafe_type_lint(ty, sp, fluent::lint_improper_ctypes_opaque, None);
            true
        } else {
            false
        }
    }

    fn check_type_for_ffi_and_report_errors(
        &mut self,
        sp: Span,
        ty: Ty<'tcx>,
        is_static: bool,
        is_return_type: bool,
    ) {
        if self.check_for_opaque_ty(sp, ty) {
            // We've already emitted an error due to an opaque type.
            return;
        }

        let ty = self.cx.tcx.try_normalize_erasing_regions(self.cx.typing_env(), ty).unwrap_or(ty);

        // C doesn't really support passing arrays by value - the only way to pass an array by value
        // is through a struct. So, first test that the top level isn't an array, and then
        // recursively check the types inside.
        if !is_static && self.check_for_array_ty(sp, ty) {
            return;
        }

        // Don't report FFI errors for unit return types. This check exists here, and not in
        // the caller (where it would make more sense) so that normalization has definitely
        // happened.
        if is_return_type && ty.is_unit() {
            return;
        }

        let mut acc = CTypesVisitorState { cache: FxHashSet::default(), base_ty: ty };
        match self.check_type_for_ffi(&mut acc, ty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(
                    ty,
                    sp,
                    fluent::lint_improper_ctypes_only_phantomdata,
                    None,
                );
            }
            FfiResult::FfiUnsafe { ty, reason, help } => {
                self.emit_ffi_unsafe_type_lint(ty, sp, reason, help);
            }
        }
    }

    /// Check if a function's argument types and result type are "ffi-safe".
    ///
    /// For a external ABI function, argument types and the result type are walked to find fn-ptr
    /// types that have external ABIs, as these still need checked.
    fn check_fn(&mut self, def_id: LocalDefId, decl: &'tcx hir::FnDecl<'_>) {
        let sig = self.cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = self.cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            for (fn_ptr_ty, span) in self.find_fn_ptr_ty_with_external_abi(input_hir, *input_ty) {
                self.check_type_for_ffi_and_report_errors(span, fn_ptr_ty, false, false);
            }
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            for (fn_ptr_ty, span) in self.find_fn_ptr_ty_with_external_abi(ret_hir, sig.output()) {
                self.check_type_for_ffi_and_report_errors(span, fn_ptr_ty, false, true);
            }
        }
    }

    /// Check if a function's argument types and result type are "ffi-safe".
    fn check_foreign_fn(&mut self, def_id: LocalDefId, decl: &'tcx hir::FnDecl<'_>) {
        let sig = self.cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = self.cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.span, *input_ty, false, false);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            self.check_type_for_ffi_and_report_errors(ret_hir.span, sig.output(), false, true);
        }
    }

    fn check_foreign_static(&mut self, id: hir::OwnerId, span: Span) {
        let ty = self.cx.tcx.type_of(id).instantiate_identity();
        self.check_type_for_ffi_and_report_errors(span, ty, true, false);
    }

    /// Find any fn-ptr types with external ABIs in `ty`.
    ///
    /// For example, `Option<extern "C" fn()>` returns `extern "C" fn()`
    fn find_fn_ptr_ty_with_external_abi(
        &self,
        hir_ty: &hir::Ty<'tcx>,
        ty: Ty<'tcx>,
    ) -> Vec<(Ty<'tcx>, Span)> {
        struct FnPtrFinder<'tcx> {
            spans: Vec<Span>,
            tys: Vec<Ty<'tcx>>,
        }

        impl<'tcx> hir::intravisit::Visitor<'_> for FnPtrFinder<'tcx> {
            fn visit_ty(&mut self, ty: &'_ hir::Ty<'_, AmbigArg>) {
                debug!(?ty);
                if let hir::TyKind::BareFn(hir::BareFnTy { abi, .. }) = ty.kind
                    && !abi.is_rustic_abi()
                {
                    self.spans.push(ty.span);
                }

                hir::intravisit::walk_ty(self, ty)
            }
        }

        impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for FnPtrFinder<'tcx> {
            type Result = ();

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if let ty::FnPtr(_, hdr) = ty.kind()
                    && !hdr.abi.is_rustic_abi()
                {
                    self.tys.push(ty);
                }

                ty.super_visit_with(self)
            }
        }

        let mut visitor = FnPtrFinder { spans: Vec::new(), tys: Vec::new() };
        ty.visit_with(&mut visitor);
        visitor.visit_ty_unambig(hir_ty);

        iter::zip(visitor.tys.drain(..), visitor.spans.drain(..)).collect()
    }
}

impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDeclarations {
    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, it: &hir::ForeignItem<'tcx>) {
        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Declaration };
        let abi = cx.tcx.hir_get_foreign_abi(it.hir_id());

        match it.kind {
            hir::ForeignItemKind::Fn(sig, _, _) => {
                if abi.is_rustic_abi() {
                    vis.check_fn(it.owner_id.def_id, sig.decl)
                } else {
                    vis.check_foreign_fn(it.owner_id.def_id, sig.decl);
                }
            }
            hir::ForeignItemKind::Static(ty, _, _) if !abi.is_rustic_abi() => {
                vis.check_foreign_static(it.owner_id, ty.span);
            }
            hir::ForeignItemKind::Static(..) | hir::ForeignItemKind::Type => (),
        }
    }
}

impl ImproperCTypesDefinitions {
    fn check_ty_maybe_containing_foreign_fnptr<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        hir_ty: &'tcx hir::Ty<'_>,
        ty: Ty<'tcx>,
    ) {
        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Definition };
        for (fn_ptr_ty, span) in vis.find_fn_ptr_ty_with_external_abi(hir_ty, ty) {
            vis.check_type_for_ffi_and_report_errors(span, fn_ptr_ty, true, false);
        }
    }

    fn check_arg_for_power_alignment<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        // Structs (under repr(C)) follow the power alignment rule if:
        //   - the first field of the struct is a floating-point type that
        //     is greater than 4-bytes, or
        //   - the first field of the struct is an aggregate whose
        //     recursively first field is a floating-point type greater than
        //     4 bytes.
        if cx.tcx.sess.target.os != "aix" {
            return false;
        }
        if ty.is_floating_point() && ty.primitive_size(cx.tcx).bytes() > 4 {
            return true;
        } else if let Adt(adt_def, _) = ty.kind()
            && adt_def.is_struct()
            && adt_def.repr().c()
            && !adt_def.repr().packed()
            && adt_def.repr().align.is_none()
        {
            let struct_variant = adt_def.variant(VariantIdx::ZERO);
            // Within a nested struct, all fields are examined to correctly
            // report if any fields after the nested struct within the
            // original struct are misaligned.
            for struct_field in &struct_variant.fields {
                let field_ty = cx.tcx.type_of(struct_field.did).instantiate_identity();
                if self.check_arg_for_power_alignment(cx, field_ty) {
                    return true;
                }
            }
        }
        return false;
    }

    fn check_struct_for_power_alignment<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>,
    ) {
        let adt_def = cx.tcx.adt_def(item.owner_id.to_def_id());
        // repr(C) structs also with packed or aligned representation
        // should be ignored.
        if adt_def.repr().c()
            && !adt_def.repr().packed()
            && adt_def.repr().align.is_none()
            && cx.tcx.sess.target.os == "aix"
            && !adt_def.all_fields().next().is_none()
        {
            let struct_variant_data = item.expect_struct().1;
            for (index, ..) in struct_variant_data.fields().iter().enumerate() {
                // Struct fields (after the first field) are checked for the
                // power alignment rule, as fields after the first are likely
                // to be the fields that are misaligned.
                if index != 0 {
                    let first_field_def = struct_variant_data.fields()[index];
                    let def_id = first_field_def.def_id;
                    let ty = cx.tcx.type_of(def_id).instantiate_identity();
                    if self.check_arg_for_power_alignment(cx, ty) {
                        cx.emit_span_lint(
                            USES_POWER_ALIGNMENT,
                            first_field_def.span,
                            UsesPowerAlignment,
                        );
                    }
                }
            }
        }
    }
}

/// `ImproperCTypesDefinitions` checks items outside of foreign items (e.g. stuff that isn't in
/// `extern "C" { }` blocks):
///
/// - `extern "<abi>" fn` definitions are checked in the same way as the
///   `ImproperCtypesDeclarations` visitor checks functions if `<abi>` is external (e.g. "C").
/// - All other items which contain types (e.g. other functions, struct definitions, etc) are
///   checked for extern fn-ptrs with external ABIs.
impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDefinitions {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::Static(_, ty, ..)
            | hir::ItemKind::Const(_, ty, ..)
            | hir::ItemKind::TyAlias(_, ty, ..) => {
                self.check_ty_maybe_containing_foreign_fnptr(
                    cx,
                    ty,
                    cx.tcx.type_of(item.owner_id).instantiate_identity(),
                );
            }
            // See `check_fn`..
            hir::ItemKind::Fn { .. } => {}
            // Structs are checked based on if they follow the power alignment
            // rule (under repr(C)).
            hir::ItemKind::Struct(..) => {
                self.check_struct_for_power_alignment(cx, item);
            }
            // See `check_field_def`..
            hir::ItemKind::Union(..) | hir::ItemKind::Enum(..) => {}
            // Doesn't define something that can contain a external type to be checked.
            hir::ItemKind::Impl(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::GlobalAsm { .. }
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::ExternCrate(..) => {}
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx hir::FieldDef<'tcx>) {
        self.check_ty_maybe_containing_foreign_fnptr(
            cx,
            field.ty,
            cx.tcx.type_of(field.def_id).instantiate_identity(),
        );
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        _: &'tcx hir::Body<'_>,
        _: Span,
        id: LocalDefId,
    ) {
        use hir::intravisit::FnKind;

        let abi = match kind {
            FnKind::ItemFn(_, _, header, ..) => header.abi,
            FnKind::Method(_, sig, ..) => sig.header.abi,
            _ => return,
        };

        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Definition };
        if abi.is_rustic_abi() {
            vis.check_fn(id, decl);
        } else {
            vis.check_foreign_fn(id, decl);
        }
    }
}

declare_lint_pass!(VariantSizeDifferences => [VARIANT_SIZE_DIFFERENCES]);

impl<'tcx> LateLintPass<'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Enum(_, ref enum_definition, _) = it.kind {
            let t = cx.tcx.type_of(it.owner_id).instantiate_identity();
            let ty = cx.tcx.erase_regions(t);
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
            && let Some(impl_did) = cx.tcx.impl_of_method(m_def_id)
            && let Some(adt) = cx.tcx.type_of(impl_did).instantiate_identity().ty_adt_def()
            // skip extension traits, only lint functions from the standard library
            && cx.tcx.trait_id_of_impl(impl_did).is_none()
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
