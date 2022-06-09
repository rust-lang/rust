use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_parent_expr, match_def_path, paths, SpanlessEq};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::ExprKind::Assign;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

const ACCEPTABLE_METHODS: [&[&str]; 4] = [
    &paths::HASHSET_ITER,
    &paths::BTREESET_ITER,
    &paths::SLICE_INTO,
    &paths::VEC_DEQUE_ITER,
];
const ACCEPTABLE_TYPES: [rustc_span::Symbol; 6] = [
    sym::BTreeSet,
    sym::BTreeMap,
    sym::HashSet,
    sym::HashMap,
    sym::Vec,
    sym::VecDeque,
];

declare_clippy_lint! {
    /// ### What it does
    /// Checks for code to be replaced by `.retain()`.
    /// ### Why is this bad?
    /// `.retain()` is simpler and avoids needless allocation.
    /// ### Example
    /// ```rust
    /// let mut vec = vec![0, 1, 2];
    /// vec = vec.iter().filter(|&x| x % 2 == 0).copied().collect();
    /// vec = vec.into_iter().filter(|x| x % 2 == 0).collect();
    /// ```
    /// Use instead:
    /// ```rust
    /// let mut vec = vec![0, 1, 2];
    /// vec.retain(|x| x % 2 == 0);
    /// ```
    #[clippy::version = "1.63.0"]
    pub USE_RETAIN,
    style,
    "`retain()` is simpler and the same functionalitys"
}
declare_lint_pass!(UseRetain => [USE_RETAIN]);

impl<'tcx> LateLintPass<'tcx> for UseRetain {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            if let Some(parent_expr) = get_parent_expr(cx, expr);
            if let Assign(left_expr, collect_expr, _) = &parent_expr.kind;
            if let hir::ExprKind::MethodCall(seg, _, _) = &collect_expr.kind;
            if seg.args.is_none();

            if let hir::ExprKind::MethodCall(_, [target_expr], _) = &collect_expr.kind;
            if let Some(collect_def_id) = cx.typeck_results().type_dependent_def_id(collect_expr.hir_id);
            if match_def_path(cx, collect_def_id, &paths::CORE_ITER_COLLECT);

            then {
                check_into_iter(cx, parent_expr, left_expr, target_expr);
                check_iter(cx, parent_expr, left_expr, target_expr);
                check_to_owned(cx, parent_expr, left_expr, target_expr);
            }
        }
    }
}

fn check_into_iter(
    cx: &LateContext<'_>,
    parent_expr: &hir::Expr<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
) {
    if_chain! {
        if let hir::ExprKind::MethodCall(_, [into_iter_expr, _], _) = &target_expr.kind;
        if let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id);
        if match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER);

        if let hir::ExprKind::MethodCall(_, [struct_expr], _) = &into_iter_expr.kind;
        if let Some(into_iter_def_id) = cx.typeck_results().type_dependent_def_id(into_iter_expr.hir_id);
        if match_def_path(cx, into_iter_def_id, &paths::CORE_ITER_INTO_ITER);
        if match_acceptable_type(cx, left_expr);

        if SpanlessEq::new(cx).eq_expr(left_expr, struct_expr);

        then {
            suggest(cx, parent_expr, left_expr, target_expr);
        }
    }
}

fn check_iter(
    cx: &LateContext<'_>,
    parent_expr: &hir::Expr<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
) {
    if_chain! {
        if let hir::ExprKind::MethodCall(_, [filter_expr], _) = &target_expr.kind;
        if let Some(copied_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id);
        if match_def_path(cx, copied_def_id, &paths::CORE_ITER_COPIED);

        if let hir::ExprKind::MethodCall(_, [iter_expr, _], _) = &filter_expr.kind;
        if let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(filter_expr.hir_id);
        if match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER);

        if let hir::ExprKind::MethodCall(_, [struct_expr], _) = &iter_expr.kind;
        if let Some(iter_expr_def_id) = cx.typeck_results().type_dependent_def_id(iter_expr.hir_id);
        if match_acceptable_def_path(cx, iter_expr_def_id);
        if match_acceptable_type(cx, left_expr);
        if SpanlessEq::new(cx).eq_expr(left_expr, struct_expr);

        then {
            suggest(cx, parent_expr, left_expr, filter_expr);
        }
    }
}

fn check_to_owned(
    cx: &LateContext<'_>,
    parent_expr: &hir::Expr<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
) {
    if_chain! {
        if let hir::ExprKind::MethodCall(_, [filter_expr], _) = &target_expr.kind;
        if let Some(to_owned_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id);
        if match_def_path(cx, to_owned_def_id, &paths::TO_OWNED_METHOD);

        if let hir::ExprKind::MethodCall(_, [chars_expr, _], _) = &filter_expr.kind;
        if let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(filter_expr.hir_id);
        if match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER);

        if let hir::ExprKind::MethodCall(_, [str_expr], _) = &chars_expr.kind;
        if let Some(chars_expr_def_id) = cx.typeck_results().type_dependent_def_id(chars_expr.hir_id);
        if match_def_path(cx, chars_expr_def_id, &paths::STR_CHARS);

        let ty = cx.typeck_results().expr_ty(str_expr).peel_refs();
        if is_type_diagnostic_item(cx, ty, sym::String);
        if SpanlessEq::new(cx).eq_expr(left_expr, str_expr);

        then {
            suggest(cx, parent_expr, left_expr, filter_expr);
        }
    }
}

fn suggest(cx: &LateContext<'_>, parent_expr: &hir::Expr<'_>, left_expr: &hir::Expr<'_>, filter_expr: &hir::Expr<'_>) {
    if_chain! {
        if let hir::ExprKind::MethodCall(_, [_, closure], _) = filter_expr.kind;
        if let hir::ExprKind::Closure(_, _, filter_body_id, ..) = closure.kind;
        let filter_body = cx.tcx.hir().body(filter_body_id);
        if let [filter_params] = filter_body.params;
        if let Some(sugg) = match filter_params.pat.kind {
            hir::PatKind::Binding(_, _, filter_param_ident, None) => {
                Some(format!("{}.retain(|{}| {})", snippet(cx, left_expr.span, ".."), filter_param_ident, snippet(cx, filter_body.value.span, "..")))
            },
            hir::PatKind::Tuple([key_pat, value_pat], _) => {
                make_sugg(cx, key_pat, value_pat, left_expr, filter_body)
            },
            hir::PatKind::Ref(pat, _) => {
                match pat.kind {
                    hir::PatKind::Binding(_, _, filter_param_ident, None) => {
                        Some(format!("{}.retain(|{}| {})", snippet(cx, left_expr.span, ".."), filter_param_ident, snippet(cx, filter_body.value.span, "..")))
                    },
                    _ => None
                }
            },
            _ => None
        };
        then {
            span_lint_and_sugg(
                cx,
                USE_RETAIN,
                parent_expr.span,
                "this expression can be written more simply using `.retain()`",
                "consider calling `.retain()` instead",
                sugg,
                Applicability::MachineApplicable
            );
        }
    }
}

fn make_sugg(
    cx: &LateContext<'_>,
    key_pat: &rustc_hir::Pat<'_>,
    value_pat: &rustc_hir::Pat<'_>,
    left_expr: &hir::Expr<'_>,
    filter_body: &hir::Body<'_>,
) -> Option<String> {
    match (&key_pat.kind, &value_pat.kind) {
        (hir::PatKind::Binding(_, _, key_param_ident, None), hir::PatKind::Binding(_, _, value_param_ident, None)) => {
            Some(format!(
                "{}.retain(|{}, &mut {}| {})",
                snippet(cx, left_expr.span, ".."),
                key_param_ident,
                value_param_ident,
                snippet(cx, filter_body.value.span, "..")
            ))
        },
        (hir::PatKind::Binding(_, _, key_param_ident, None), hir::PatKind::Wild) => Some(format!(
            "{}.retain(|{}, _| {})",
            snippet(cx, left_expr.span, ".."),
            key_param_ident,
            snippet(cx, filter_body.value.span, "..")
        )),
        (hir::PatKind::Wild, hir::PatKind::Binding(_, _, value_param_ident, None)) => Some(format!(
            "{}.retain(|_, &mut {}| {})",
            snippet(cx, left_expr.span, ".."),
            value_param_ident,
            snippet(cx, filter_body.value.span, "..")
        )),
        _ => None,
    }
}

fn match_acceptable_def_path(cx: &LateContext<'_>, collect_def_id: DefId) -> bool {
    return ACCEPTABLE_METHODS
        .iter()
        .any(|&method| match_def_path(cx, collect_def_id, method));
}

fn match_acceptable_type(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr).peel_refs();
    return ACCEPTABLE_TYPES
        .iter()
        .any(|&ty| is_type_diagnostic_item(cx, expr_ty, ty));
}
