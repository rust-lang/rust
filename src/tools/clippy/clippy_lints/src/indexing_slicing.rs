use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::ty::{deref_chain, get_adt_inherent_method};
use clippy_utils::{higher, is_from_proc_macro, is_in_test, sym};
use rustc_ast::ast::RangeLimits;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for out of bounds array indexing with a constant
    /// index.
    ///
    /// ### Why is this bad?
    /// This will always panic at runtime.
    ///
    /// ### Example
    /// ```rust,no_run
    /// let x = [1, 2, 3, 4];
    ///
    /// x[9];
    /// &x[2..9];
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = [1, 2, 3, 4];
    /// // Index within bounds
    ///
    /// x[0];
    /// x[3];
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OUT_OF_BOUNDS_INDEXING,
    correctness,
    "out of bounds constant indexing"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of indexing or slicing that may panic at runtime.
    ///
    /// This lint does not report on indexing or slicing operations
    /// that always panic, clippy's `out_of_bound_indexing` already
    /// handles those cases.
    ///
    /// ### Why restrict this?
    /// To avoid implicit panics from indexing and slicing.
    ///
    /// There are “checked” alternatives which do not panic, and can be used with `unwrap()` to make
    /// an explicit panic when it is desired.
    ///
    /// ### Limitations
    /// This lint does not check for the usage of indexing or slicing on strings. These are covered
    /// by the more specific `string_slice` lint.
    ///
    /// ### Example
    /// ```rust,no_run
    /// // Vector
    /// let x = vec![0, 1, 2, 3];
    ///
    /// x[2];
    /// x[100];
    /// &x[2..100];
    ///
    /// // Array
    /// let y = [0, 1, 2, 3];
    ///
    /// let i = 10; // Could be a runtime value
    /// let j = 20;
    /// &y[i..j];
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = vec![0, 1, 2, 3];
    /// x.get(2);
    /// x.get(100);
    /// x.get(2..100);
    ///
    /// # let y = [0, 1, 2, 3];
    /// let i = 10;
    /// let j = 20;
    /// y.get(i..j);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INDEXING_SLICING,
    restriction,
    "indexing/slicing usage"
}

impl_lint_pass!(IndexingSlicing => [INDEXING_SLICING, OUT_OF_BOUNDS_INDEXING]);

pub struct IndexingSlicing {
    allow_indexing_slicing_in_tests: bool,
    suppress_restriction_lint_in_const: bool,
}

impl IndexingSlicing {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            allow_indexing_slicing_in_tests: conf.allow_indexing_slicing_in_tests,
            suppress_restriction_lint_in_const: conf.suppress_restriction_lint_in_const,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for IndexingSlicing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Index(array, index, _) = &expr.kind
            && (!self.suppress_restriction_lint_in_const || !cx.tcx.hir_is_inside_const_context(expr.hir_id))
            && let expr_ty = cx.typeck_results().expr_ty(array)
            && let mut deref = deref_chain(cx, expr_ty)
            && deref.any(|l| {
                l.peel_refs().is_slice()
                    || l.peel_refs().is_array()
                    || ty_has_applicable_get_function(cx, l.peel_refs(), expr_ty, expr)
            })
            && !is_from_proc_macro(cx, expr)
        {
            let note = "the suggestion might not be applicable in constant blocks";
            let ty = cx.typeck_results().expr_ty(array).peel_refs();
            let allowed_in_tests = self.allow_indexing_slicing_in_tests && is_in_test(cx.tcx, expr.hir_id);
            if let Some(range) = higher::Range::hir(index) {
                // Ranged indexes, i.e., &x[n..m], &x[n..], &x[..n] and &x[..]
                if let ty::Array(_, s) = ty.kind() {
                    let size: u128 = if let Some(size) = s.try_to_target_usize(cx.tcx) {
                        size.into()
                    } else {
                        return;
                    };

                    let const_range = to_const_range(cx, range, size);

                    if let (Some(start), _) = const_range
                        && start > size
                    {
                        span_lint(
                            cx,
                            OUT_OF_BOUNDS_INDEXING,
                            range.start.map_or(expr.span, |start| start.span),
                            "range is out of bounds",
                        );
                        return;
                    }

                    if let (_, Some(end)) = const_range
                        && end > size
                    {
                        span_lint(
                            cx,
                            OUT_OF_BOUNDS_INDEXING,
                            range.end.map_or(expr.span, |end| end.span),
                            "range is out of bounds",
                        );
                        return;
                    }

                    if let (Some(_), Some(_)) = const_range {
                        // early return because both start and end are constants
                        // and we have proven above that they are in bounds
                        return;
                    }
                }

                let help_msg = match (range.start, range.end) {
                    (None, Some(_)) => "consider using `.get(..n)`or `.get_mut(..n)` instead",
                    (Some(_), None) => "consider using `.get(n..)` or .get_mut(n..)` instead",
                    (Some(_), Some(_)) => "consider using `.get(n..m)` or `.get_mut(n..m)` instead",
                    (None, None) => return, // [..] is ok.
                };

                if allowed_in_tests {
                    return;
                }

                span_lint_and_then(cx, INDEXING_SLICING, expr.span, "slicing may panic", |diag| {
                    diag.help(help_msg);

                    if cx.tcx.hir_is_inside_const_context(expr.hir_id) {
                        diag.note(note);
                    }
                });
            } else {
                // Catchall non-range index, i.e., [n] or [n << m]
                if let ty::Array(..) = ty.kind() {
                    // Index is a const block.
                    if let ExprKind::ConstBlock(..) = index.kind {
                        return;
                    }
                    // Index is a constant uint.
                    if let Some(constant) = ConstEvalCtxt::new(cx).eval(index) {
                        // only `usize` index is legal in rust array index
                        // leave other type to rustc
                        if let Constant::Int(off) = constant
                            && off <= usize::MAX as u128
                            && let ty::Uint(utype) = cx.typeck_results().expr_ty(index).kind()
                            && *utype == ty::UintTy::Usize
                            && let ty::Array(_, s) = ty.kind()
                            && let Some(size) = s.try_to_target_usize(cx.tcx)
                        {
                            // get constant offset and check whether it is in bounds
                            let off = usize::try_from(off).unwrap();
                            let size = usize::try_from(size).unwrap();

                            if off >= size {
                                span_lint(cx, OUT_OF_BOUNDS_INDEXING, expr.span, "index is out of bounds");
                            }
                        }
                        // Let rustc's `const_err` lint handle constant `usize` indexing on arrays.
                        return;
                    }
                }

                if allowed_in_tests {
                    return;
                }

                span_lint_and_then(cx, INDEXING_SLICING, expr.span, "indexing may panic", |diag| {
                    diag.help("consider using `.get(n)` or `.get_mut(n)` instead");

                    if cx.tcx.hir_is_inside_const_context(expr.hir_id) {
                        diag.note(note);
                    }
                });
            }
        }
    }
}

/// Returns a tuple of options with the start and end (exclusive) values of
/// the range. If the start or end is not constant, None is returned.
fn to_const_range(cx: &LateContext<'_>, range: higher::Range<'_>, array_size: u128) -> (Option<u128>, Option<u128>) {
    let ecx = ConstEvalCtxt::new(cx);
    let s = range.start.map(|expr| ecx.eval(expr));
    let start = match s {
        Some(Some(Constant::Int(x))) => Some(x),
        Some(_) => None,
        None => Some(0),
    };

    let e = range.end.map(|expr| ecx.eval(expr));
    let end = match e {
        Some(Some(Constant::Int(x))) => {
            if range.limits == RangeLimits::Closed {
                Some(x + 1)
            } else {
                Some(x)
            }
        },
        Some(_) => None,
        None => Some(array_size),
    };

    (start, end)
}

/// Checks if the output Ty of the `get` method on this Ty (if any) matches the Ty returned by the
/// indexing operation (if any).
fn ty_has_applicable_get_function<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
    array_ty: Ty<'tcx>,
    index_expr: &Expr<'_>,
) -> bool {
    if let ty::Adt(_, _) = array_ty.kind()
        && let Some(get_output_ty) = get_adt_inherent_method(cx, ty, sym::get).map(|m| {
            cx.tcx
                .fn_sig(m.def_id)
                .skip_binder()
                .output()
                .skip_binder()
        })
        && let ty::Adt(def, args) = get_output_ty.kind()
        && cx.tcx.is_diagnostic_item(sym::Option, def.0.did)
        && let Some(option_generic_param) = args.first()
        && let generic_ty = option_generic_param.expect_ty().peel_refs()
        // FIXME: ideally this would handle type params and projections properly, for now just assume it's the same type
        && (cx.typeck_results().expr_ty(index_expr).peel_refs() == generic_ty.peel_refs()
            || matches!(generic_ty.peel_refs().kind(), ty::Param(_) | ty::Alias(_, _)))
    {
        true
    } else {
        false
    }
}
