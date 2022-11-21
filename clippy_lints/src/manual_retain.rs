use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_parent_expr, match_def_path, paths, SpanlessEq};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::ExprKind::Assign;
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;

const ACCEPTABLE_METHODS: [&[&str]; 4] = [
    &paths::HASHSET_ITER,
    &paths::BTREESET_ITER,
    &paths::SLICE_INTO,
    &paths::VEC_DEQUE_ITER,
];
const ACCEPTABLE_TYPES: [(rustc_span::Symbol, Option<RustcVersion>); 6] = [
    (sym::BTreeSet, Some(msrvs::BTREE_SET_RETAIN)),
    (sym::BTreeMap, Some(msrvs::BTREE_MAP_RETAIN)),
    (sym::HashSet, Some(msrvs::HASH_SET_RETAIN)),
    (sym::HashMap, Some(msrvs::HASH_MAP_RETAIN)),
    (sym::Vec, None),
    (sym::VecDeque, None),
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
    #[clippy::version = "1.64.0"]
    pub MANUAL_RETAIN,
    perf,
    "`retain()` is simpler and the same functionalitys"
}

pub struct ManualRetain {
    msrv: Msrv,
}

impl ManualRetain {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualRetain => [MANUAL_RETAIN]);

impl<'tcx> LateLintPass<'tcx> for ManualRetain {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let Some(parent_expr) = get_parent_expr(cx, expr)
            && let Assign(left_expr, collect_expr, _) = &parent_expr.kind
            && let hir::ExprKind::MethodCall(seg, ..) = &collect_expr.kind
            && seg.args.is_none()
            && let hir::ExprKind::MethodCall(_, target_expr, [], _) = &collect_expr.kind
            && let Some(collect_def_id) = cx.typeck_results().type_dependent_def_id(collect_expr.hir_id)
            && match_def_path(cx, collect_def_id, &paths::CORE_ITER_COLLECT) {
            check_into_iter(cx, parent_expr, left_expr, target_expr, &self.msrv);
            check_iter(cx, parent_expr, left_expr, target_expr, &self.msrv);
            check_to_owned(cx, parent_expr, left_expr, target_expr, &self.msrv);
        }
    }

    extract_msrv_attr!(LateContext);
}

fn check_into_iter(
    cx: &LateContext<'_>,
    parent_expr: &hir::Expr<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
    msrv: &Msrv,
) {
    if let hir::ExprKind::MethodCall(_, into_iter_expr, [_], _) = &target_expr.kind
        && let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id)
        && match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER)
        && let hir::ExprKind::MethodCall(_, struct_expr, [], _) = &into_iter_expr.kind
        && let Some(into_iter_def_id) = cx.typeck_results().type_dependent_def_id(into_iter_expr.hir_id)
        && cx.tcx.lang_items().require(hir::LangItem::IntoIterIntoIter).ok() == Some(into_iter_def_id)
        && match_acceptable_type(cx, left_expr, msrv)
        && SpanlessEq::new(cx).eq_expr(left_expr, struct_expr) {
        suggest(cx, parent_expr, left_expr, target_expr);
    }
}

fn check_iter(
    cx: &LateContext<'_>,
    parent_expr: &hir::Expr<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
    msrv: &Msrv,
) {
    if let hir::ExprKind::MethodCall(_, filter_expr, [], _) = &target_expr.kind
        && let Some(copied_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id)
        && (match_def_path(cx, copied_def_id, &paths::CORE_ITER_COPIED)
            || match_def_path(cx, copied_def_id, &paths::CORE_ITER_CLONED))
        && let hir::ExprKind::MethodCall(_, iter_expr, [_], _) = &filter_expr.kind
        && let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(filter_expr.hir_id)
        && match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER)
        && let hir::ExprKind::MethodCall(_, struct_expr, [], _) = &iter_expr.kind
        && let Some(iter_expr_def_id) = cx.typeck_results().type_dependent_def_id(iter_expr.hir_id)
        && match_acceptable_def_path(cx, iter_expr_def_id)
        && match_acceptable_type(cx, left_expr, msrv)
        && SpanlessEq::new(cx).eq_expr(left_expr, struct_expr) {
        suggest(cx, parent_expr, left_expr, filter_expr);
    }
}

fn check_to_owned(
    cx: &LateContext<'_>,
    parent_expr: &hir::Expr<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
    msrv: &Msrv,
) {
    if msrv.meets(msrvs::STRING_RETAIN)
        && let hir::ExprKind::MethodCall(_, filter_expr, [], _) = &target_expr.kind
        && let Some(to_owned_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id)
        && match_def_path(cx, to_owned_def_id, &paths::TO_OWNED_METHOD)
        && let hir::ExprKind::MethodCall(_, chars_expr, [_], _) = &filter_expr.kind
        && let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(filter_expr.hir_id)
        && match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER)
        && let hir::ExprKind::MethodCall(_, str_expr, [], _) = &chars_expr.kind
        && let Some(chars_expr_def_id) = cx.typeck_results().type_dependent_def_id(chars_expr.hir_id)
        && match_def_path(cx, chars_expr_def_id, &paths::STR_CHARS)
        && let ty = cx.typeck_results().expr_ty(str_expr).peel_refs()
        && is_type_diagnostic_item(cx, ty, sym::String)
        && SpanlessEq::new(cx).eq_expr(left_expr, str_expr) {
        suggest(cx, parent_expr, left_expr, filter_expr);
    }
}

fn suggest(cx: &LateContext<'_>, parent_expr: &hir::Expr<'_>, left_expr: &hir::Expr<'_>, filter_expr: &hir::Expr<'_>) {
    if let hir::ExprKind::MethodCall(_, _, [closure], _) = filter_expr.kind
        && let hir::ExprKind::Closure(&hir::Closure { body, ..}) = closure.kind
        && let filter_body = cx.tcx.hir().body(body)
        && let [filter_params] = filter_body.params
        && let Some(sugg) = match filter_params.pat.kind {
            hir::PatKind::Binding(_, _, filter_param_ident, None) => {
                Some(format!("{}.retain(|{filter_param_ident}| {})", snippet(cx, left_expr.span, ".."), snippet(cx, filter_body.value.span, "..")))
            },
            hir::PatKind::Tuple([key_pat, value_pat], _) => {
                make_sugg(cx, key_pat, value_pat, left_expr, filter_body)
            },
            hir::PatKind::Ref(pat, _) => {
                match pat.kind {
                    hir::PatKind::Binding(_, _, filter_param_ident, None) => {
                        Some(format!("{}.retain(|{filter_param_ident}| {})", snippet(cx, left_expr.span, ".."), snippet(cx, filter_body.value.span, "..")))
                    },
                    _ => None
                }
            },
            _ => None
        } {
        span_lint_and_sugg(
            cx,
            MANUAL_RETAIN,
            parent_expr.span,
            "this expression can be written more simply using `.retain()`",
            "consider calling `.retain()` instead",
            sugg,
            Applicability::MachineApplicable
        );
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
                "{}.retain(|{key_param_ident}, &mut {value_param_ident}| {})",
                snippet(cx, left_expr.span, ".."),
                snippet(cx, filter_body.value.span, "..")
            ))
        },
        (hir::PatKind::Binding(_, _, key_param_ident, None), hir::PatKind::Wild) => Some(format!(
            "{}.retain(|{key_param_ident}, _| {})",
            snippet(cx, left_expr.span, ".."),
            snippet(cx, filter_body.value.span, "..")
        )),
        (hir::PatKind::Wild, hir::PatKind::Binding(_, _, value_param_ident, None)) => Some(format!(
            "{}.retain(|_, &mut {value_param_ident}| {})",
            snippet(cx, left_expr.span, ".."),
            snippet(cx, filter_body.value.span, "..")
        )),
        _ => None,
    }
}

fn match_acceptable_def_path(cx: &LateContext<'_>, collect_def_id: DefId) -> bool {
    ACCEPTABLE_METHODS
        .iter()
        .any(|&method| match_def_path(cx, collect_def_id, method))
}

fn match_acceptable_type(cx: &LateContext<'_>, expr: &hir::Expr<'_>, msrv: &Msrv) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr).peel_refs();
    ACCEPTABLE_TYPES.iter().any(|(ty, acceptable_msrv)| {
        is_type_diagnostic_item(cx, expr_ty, *ty)
            && acceptable_msrv.map_or(true, |acceptable_msrv| msrv.meets(acceptable_msrv))
    })
}
