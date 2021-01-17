use crate::consts::{constant, Constant};
use crate::utils::usage::mutated_variables;
use crate::utils::{
    eq_expr_value, higher, match_def_path, meets_msrv, multispan_sugg, paths, qpath_res, snippet, span_lint_and_then,
};

use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::BinOpKind;
use rustc_hir::{BorrowKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::ty;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Spanned;
use rustc_span::Span;

const MANUAL_STRIP_MSRV: RustcVersion = RustcVersion::new(1, 45, 0);

declare_clippy_lint! {
    /// **What it does:**
    /// Suggests using `strip_{prefix,suffix}` over `str::{starts,ends}_with` and slicing using
    /// the pattern's length.
    ///
    /// **Why is this bad?**
    /// Using `str:strip_{prefix,suffix}` is safer and may have better performance as there is no
    /// slicing which may panic and the compiler does not need to insert this panic code. It is
    /// also sometimes more readable as it removes the need for duplicating or storing the pattern
    /// used by `str::{starts,ends}_with` and in the slicing.
    ///
    /// **Known problems:**
    /// None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let s = "hello, world!";
    /// if s.starts_with("hello, ") {
    ///     assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let s = "hello, world!";
    /// if let Some(end) = s.strip_prefix("hello, ") {
    ///     assert_eq!(end.to_uppercase(), "WORLD!");
    /// }
    /// ```
    pub MANUAL_STRIP,
    complexity,
    "suggests using `strip_{prefix,suffix}` over `str::{starts,ends}_with` and slicing"
}

pub struct ManualStrip {
    msrv: Option<RustcVersion>,
}

impl ManualStrip {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualStrip => [MANUAL_STRIP]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StripKind {
    Prefix,
    Suffix,
}

impl<'tcx> LateLintPass<'tcx> for ManualStrip {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &MANUAL_STRIP_MSRV) {
            return;
        }

        if_chain! {
            if let ExprKind::If(cond, then, _) = &expr.kind;
            if let ExprKind::MethodCall(_, _, [target_arg, pattern], _) = cond.kind;
            if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(cond.hir_id);
            if let ExprKind::Path(target_path) = &target_arg.kind;
            then {
                let strip_kind = if match_def_path(cx, method_def_id, &paths::STR_STARTS_WITH) {
                    StripKind::Prefix
                } else if match_def_path(cx, method_def_id, &paths::STR_ENDS_WITH) {
                    StripKind::Suffix
                } else {
                    return;
                };
                let target_res = qpath_res(cx, &target_path, target_arg.hir_id);
                if target_res == Res::Err {
                    return;
                };

                if_chain! {
                    if let Res::Local(hir_id) = target_res;
                    if let Some(used_mutably) = mutated_variables(then, cx);
                    if used_mutably.contains(&hir_id);
                    then {
                        return;
                    }
                }

                let strippings = find_stripping(cx, strip_kind, target_res, pattern, then);
                if !strippings.is_empty() {

                    let kind_word = match strip_kind {
                        StripKind::Prefix => "prefix",
                        StripKind::Suffix => "suffix",
                    };

                    let test_span = expr.span.until(then.span);
                    span_lint_and_then(cx, MANUAL_STRIP, strippings[0], &format!("stripping a {} manually", kind_word), |diag| {
                        diag.span_note(test_span, &format!("the {} was tested here", kind_word));
                        multispan_sugg(
                            diag,
                            &format!("try using the `strip_{}` method", kind_word),
                            vec![(test_span,
                                  format!("if let Some(<stripped>) = {}.strip_{}({}) ",
                                          snippet(cx, target_arg.span, ".."),
                                          kind_word,
                                          snippet(cx, pattern.span, "..")))]
                            .into_iter().chain(strippings.into_iter().map(|span| (span, "<stripped>".into()))),
                        )
                    });
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

// Returns `Some(arg)` if `expr` matches `arg.len()` and `None` otherwise.
fn len_arg<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if_chain! {
        if let ExprKind::MethodCall(_, _, [arg], _) = expr.kind;
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if match_def_path(cx, method_def_id, &paths::STR_LEN);
        then {
            Some(arg)
        } else {
            None
        }
    }
}

// Returns the length of the `expr` if it's a constant string or char.
fn constant_length(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<u128> {
    let (value, _) = constant(cx, cx.typeck_results(), expr)?;
    match value {
        Constant::Str(value) => Some(value.len() as u128),
        Constant::Char(value) => Some(value.len_utf8() as u128),
        _ => None,
    }
}

// Tests if `expr` equals the length of the pattern.
fn eq_pattern_length<'tcx>(cx: &LateContext<'tcx>, pattern: &Expr<'_>, expr: &'tcx Expr<'_>) -> bool {
    if let ExprKind::Lit(Spanned {
        node: LitKind::Int(n, _),
        ..
    }) = expr.kind
    {
        constant_length(cx, pattern).map_or(false, |length| length == n)
    } else {
        len_arg(cx, expr).map_or(false, |arg| eq_expr_value(cx, pattern, arg))
    }
}

// Tests if `expr` is a `&str`.
fn is_ref_str(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match cx.typeck_results().expr_ty_adjusted(&expr).kind() {
        ty::Ref(_, ty, _) => ty.is_str(),
        _ => false,
    }
}

// Removes the outer `AddrOf` expression if needed.
fn peel_ref<'a>(expr: &'a Expr<'_>) -> &'a Expr<'a> {
    if let ExprKind::AddrOf(BorrowKind::Ref, _, unref) = &expr.kind {
        unref
    } else {
        expr
    }
}

// Find expressions where `target` is stripped using the length of `pattern`.
// We'll suggest replacing these expressions with the result of the `strip_{prefix,suffix}`
// method.
fn find_stripping<'tcx>(
    cx: &LateContext<'tcx>,
    strip_kind: StripKind,
    target: Res,
    pattern: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
) -> Vec<Span> {
    struct StrippingFinder<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        strip_kind: StripKind,
        target: Res,
        pattern: &'tcx Expr<'tcx>,
        results: Vec<Span>,
    }

    impl<'a, 'tcx> Visitor<'tcx> for StrippingFinder<'a, 'tcx> {
        type Map = Map<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
            if_chain! {
                if is_ref_str(self.cx, ex);
                let unref = peel_ref(ex);
                if let ExprKind::Index(indexed, index) = &unref.kind;
                if let Some(higher::Range { start, end, .. }) = higher::range(index);
                if let ExprKind::Path(path) = &indexed.kind;
                if qpath_res(self.cx, path, ex.hir_id) == self.target;
                then {
                    match (self.strip_kind, start, end) {
                        (StripKind::Prefix, Some(start), None) => {
                            if eq_pattern_length(self.cx, self.pattern, start) {
                                self.results.push(ex.span);
                                return;
                            }
                        },
                        (StripKind::Suffix, None, Some(end)) => {
                            if_chain! {
                                if let ExprKind::Binary(Spanned { node: BinOpKind::Sub, .. }, left, right) = end.kind;
                                if let Some(left_arg) = len_arg(self.cx, left);
                                if let ExprKind::Path(left_path) = &left_arg.kind;
                                if qpath_res(self.cx, left_path, left_arg.hir_id) == self.target;
                                if eq_pattern_length(self.cx, self.pattern, right);
                                then {
                                    self.results.push(ex.span);
                                    return;
                                }
                            }
                        },
                        _ => {}
                    }
                }
            }

            walk_expr(self, ex);
        }
    }

    let mut finder = StrippingFinder {
        cx,
        strip_kind,
        target,
        pattern,
        results: vec![],
    };
    walk_expr(&mut finder, expr);
    finder.results
}
