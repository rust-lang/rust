use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::usage::mutated_variables;
use clippy_utils::{eq_expr_value, higher};
use rustc_ast::BindingMode;
use rustc_ast::ast::LitKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{Visitor, walk_expr, walk_pat};
use rustc_hir::{BinOpKind, BorrowKind, Expr, ExprKind, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass, LintContext as _};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::source_map::Spanned;
use rustc_span::{Symbol, sym};
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Suggests using `strip_{prefix,suffix}` over `str::{starts,ends}_with` and slicing using
    /// the pattern's length.
    ///
    /// ### Why is this bad?
    /// Using `str:strip_{prefix,suffix}` is safer and may have better performance as there is no
    /// slicing which may panic and the compiler does not need to insert this panic code. It is
    /// also sometimes more readable as it removes the need for duplicating or storing the pattern
    /// used by `str::{starts,ends}_with` and in the slicing.
    ///
    /// ### Example
    /// ```no_run
    /// let s = "hello, world!";
    /// if s.starts_with("hello, ") {
    ///     assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let s = "hello, world!";
    /// if let Some(end) = s.strip_prefix("hello, ") {
    ///     assert_eq!(end.to_uppercase(), "WORLD!");
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub MANUAL_STRIP,
    complexity,
    "suggests using `strip_{prefix,suffix}` over `str::{starts,ends}_with` and slicing"
}

pub struct ManualStrip {
    msrv: Msrv,
}

impl ManualStrip {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
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
        if let Some(higher::If { cond, then, .. }) = higher::If::hir(expr)
            && let ExprKind::MethodCall(_, target_arg, [pattern], _) = cond.kind
            && let ExprKind::Path(target_path) = &target_arg.kind
            && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(cond.hir_id)
        {
            let strip_kind = if cx.tcx.is_diagnostic_item(sym::str_starts_with, method_def_id) {
                StripKind::Prefix
            } else if cx.tcx.is_diagnostic_item(sym::str_ends_with, method_def_id) {
                StripKind::Suffix
            } else {
                return;
            };
            let target_res = cx.qpath_res(target_path, target_arg.hir_id);
            if target_res == Res::Err {
                return;
            }

            if let Res::Local(hir_id) = target_res
                && let Some(used_mutably) = mutated_variables(then, cx)
                && used_mutably.contains(&hir_id)
            {
                return;
            }

            let (strippings, bindings) = find_stripping(cx, strip_kind, target_res, pattern, then);
            if !strippings.is_empty() && self.msrv.meets(cx, msrvs::STR_STRIP_PREFIX) {
                let kind_word = match strip_kind {
                    StripKind::Prefix => "prefix",
                    StripKind::Suffix => "suffix",
                };

                let test_span = expr.span.until(then.span);

                // If the first use is a simple `let` statement, reuse its identifier in the `if let Some(…)` and
                // remove the `let` statement as long as the identifier is never bound again within the lexical
                // scope of interest.
                let (ident_name, let_stmt_span, skip, mut app) = if let Node::LetStmt(let_stmt) =
                    cx.tcx.parent_hir_node(strippings[0].hir_id)
                    && let PatKind::Binding(BindingMode::NONE, _, ident, None) = &let_stmt.pat.kind
                    && bindings.get(&ident.name) == Some(&1)
                {
                    (
                        ident.name.as_str(),
                        Some(cx.sess().source_map().span_extend_while_whitespace(let_stmt.span)),
                        1,
                        Applicability::MachineApplicable,
                    )
                } else {
                    ("<stripped>", None, 0, Applicability::HasPlaceholders)
                };

                span_lint_and_then(
                    cx,
                    MANUAL_STRIP,
                    strippings[0].span,
                    format!("stripping a {kind_word} manually"),
                    |diag| {
                        diag.span_note(test_span, format!("the {kind_word} was tested here"));
                        diag.multipart_suggestion(
                            format!("try using the `strip_{kind_word}` method"),
                            iter::once((
                                test_span,
                                format!(
                                    "if let Some({ident_name}) = {}.strip_{kind_word}({}) ",
                                    snippet_with_applicability(cx, target_arg.span, "_", &mut app),
                                    snippet_with_applicability(cx, pattern.span, "_", &mut app)
                                ),
                            ))
                            .chain(let_stmt_span.map(|span| (span, String::new())))
                            .chain(
                                strippings
                                    .into_iter()
                                    .skip(skip)
                                    .map(|expr| (expr.span, ident_name.into())),
                            )
                            .collect(),
                            app,
                        );
                    },
                );
            }
        }
    }
}

// Returns `Some(arg)` if `expr` matches `arg.len()` and `None` otherwise.
fn len_arg<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::MethodCall(_, arg, [], _) = expr.kind
        && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && cx.tcx.is_diagnostic_item(sym::str_len, method_def_id)
    {
        Some(arg)
    } else {
        None
    }
}

// Returns the length of the `expr` if it's a constant string or char.
fn constant_length(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<u128> {
    let value = ConstEvalCtxt::new(cx).eval(expr)?;
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
        constant_length(cx, pattern).is_some_and(|length| n == length)
    } else {
        len_arg(cx, expr).is_some_and(|arg| eq_expr_value(cx, pattern, arg))
    }
}

// Tests if `expr` is a `&str`.
fn is_ref_str(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match cx.typeck_results().expr_ty_adjusted(expr).kind() {
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

/// Find expressions where `target` is stripped using the length of `pattern`.
/// We'll suggest replacing these expressions with the result of the `strip_{prefix,suffix}`
/// method.
/// Also, all bindings found during the visit are counted and returned.
fn find_stripping<'tcx>(
    cx: &LateContext<'tcx>,
    strip_kind: StripKind,
    target: Res,
    pattern: &'tcx Expr<'_>,
    expr: &'tcx Expr<'tcx>,
) -> (Vec<&'tcx Expr<'tcx>>, FxHashMap<Symbol, usize>) {
    struct StrippingFinder<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        strip_kind: StripKind,
        target: Res,
        pattern: &'tcx Expr<'tcx>,
        results: Vec<&'tcx Expr<'tcx>>,
        bindings: FxHashMap<Symbol, usize>,
    }

    impl<'tcx> Visitor<'tcx> for StrippingFinder<'_, 'tcx> {
        fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
            if is_ref_str(self.cx, ex)
                && let unref = peel_ref(ex)
                && let ExprKind::Index(indexed, index, _) = &unref.kind
                && let Some(higher::Range { start, end, .. }) = higher::Range::hir(index)
                && let ExprKind::Path(path) = &indexed.kind
                && self.cx.qpath_res(path, ex.hir_id) == self.target
            {
                match (self.strip_kind, start, end) {
                    (StripKind::Prefix, Some(start), None) => {
                        if eq_pattern_length(self.cx, self.pattern, start) {
                            self.results.push(ex);
                            return;
                        }
                    },
                    (StripKind::Suffix, None, Some(end)) => {
                        if let ExprKind::Binary(
                            Spanned {
                                node: BinOpKind::Sub, ..
                            },
                            left,
                            right,
                        ) = end.kind
                            && let Some(left_arg) = len_arg(self.cx, left)
                            && let ExprKind::Path(left_path) = &left_arg.kind
                            && self.cx.qpath_res(left_path, left_arg.hir_id) == self.target
                            && eq_pattern_length(self.cx, self.pattern, right)
                        {
                            self.results.push(ex);
                            return;
                        }
                    },
                    _ => {},
                }
            }

            walk_expr(self, ex);
        }

        fn visit_pat(&mut self, pat: &'tcx rustc_hir::Pat<'tcx>) -> Self::Result {
            if let PatKind::Binding(_, _, ident, _) = pat.kind {
                *self.bindings.entry(ident.name).or_default() += 1;
            }
            walk_pat(self, pat);
        }
    }

    let mut finder = StrippingFinder {
        cx,
        strip_kind,
        target,
        pattern,
        results: vec![],
        bindings: FxHashMap::default(),
    };
    walk_expr(&mut finder, expr);
    (finder.results, finder.bindings)
}
