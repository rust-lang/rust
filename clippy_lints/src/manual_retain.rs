use clippy_config::msrvs::{self, Msrv};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item};
use clippy_utils::{get_parent_expr, match_def_path, paths, SpanlessEq};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::ExprKind::Assign;
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::impl_lint_pass;
use rustc_span::symbol::sym;
use rustc_span::Span;

const ACCEPTABLE_METHODS: [&[&str]; 5] = [
    &paths::BINARYHEAP_ITER,
    &paths::HASHSET_ITER,
    &paths::BTREESET_ITER,
    &paths::SLICE_INTO,
    &paths::VEC_DEQUE_ITER,
];
const ACCEPTABLE_TYPES: [(rustc_span::Symbol, Option<RustcVersion>); 7] = [
    (sym::BinaryHeap, Some(msrvs::BINARY_HEAP_RETAIN)),
    (sym::BTreeSet, Some(msrvs::BTREE_SET_RETAIN)),
    (sym::BTreeMap, Some(msrvs::BTREE_MAP_RETAIN)),
    (sym::HashSet, Some(msrvs::HASH_SET_RETAIN)),
    (sym::HashMap, Some(msrvs::HASH_MAP_RETAIN)),
    (sym::Vec, None),
    (sym::VecDeque, None),
];
const MAP_TYPES: [rustc_span::Symbol; 2] = [sym::BTreeMap, sym::HashMap];

declare_clippy_lint! {
    /// ### What it does
    /// Checks for code to be replaced by `.retain()`.
    /// ### Why is this bad?
    /// `.retain()` is simpler and avoids needless allocation.
    /// ### Example
    /// ```no_run
    /// let mut vec = vec![0, 1, 2];
    /// vec = vec.iter().filter(|&x| x % 2 == 0).copied().collect();
    /// vec = vec.into_iter().filter(|x| x % 2 == 0).collect();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut vec = vec![0, 1, 2];
    /// vec.retain(|x| x % 2 == 0);
    /// vec.retain(|x| x % 2 == 0);
    /// ```
    #[clippy::version = "1.64.0"]
    pub MANUAL_RETAIN,
    perf,
    "`retain()` is simpler and the same functionalities"
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
            && cx.tcx.is_diagnostic_item(sym::iterator_collect_fn, collect_def_id)
        {
            check_into_iter(cx, left_expr, target_expr, parent_expr.span, &self.msrv);
            check_iter(cx, left_expr, target_expr, parent_expr.span, &self.msrv);
            check_to_owned(cx, left_expr, target_expr, parent_expr.span, &self.msrv);
        }
    }

    extract_msrv_attr!(LateContext);
}

fn check_into_iter(
    cx: &LateContext<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
    parent_expr_span: Span,
    msrv: &Msrv,
) {
    if let hir::ExprKind::MethodCall(_, into_iter_expr, [_], _) = &target_expr.kind
        && let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id)
        && match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER)
        && let hir::ExprKind::MethodCall(_, struct_expr, [], _) = &into_iter_expr.kind
        && let Some(into_iter_def_id) = cx.typeck_results().type_dependent_def_id(into_iter_expr.hir_id)
        && Some(into_iter_def_id) == cx.tcx.lang_items().into_iter_fn()
        && match_acceptable_type(cx, left_expr, msrv)
        && SpanlessEq::new(cx).eq_expr(left_expr, struct_expr)
        && let hir::ExprKind::MethodCall(_, _, [closure_expr], _) = target_expr.kind
        && let hir::ExprKind::Closure(closure) = closure_expr.kind
        && let filter_body = cx.tcx.hir().body(closure.body)
        && let [filter_params] = filter_body.params
    {
        if match_map_type(cx, left_expr) {
            if let hir::PatKind::Tuple([key_pat, value_pat], _) = filter_params.pat.kind {
                if let Some(sugg) = make_sugg(cx, key_pat, value_pat, left_expr, filter_body) {
                    make_span_lint_and_sugg(cx, parent_expr_span, sugg);
                }
            }
            // Cannot lint other cases because `retain` requires two parameters
        } else {
            // Can always move because `retain` and `filter` have the same bound on the predicate
            // for other types
            make_span_lint_and_sugg(
                cx,
                parent_expr_span,
                format!(
                    "{}.retain({})",
                    snippet(cx, left_expr.span, ".."),
                    snippet(cx, closure_expr.span, "..")
                ),
            );
        }
    }
}

fn check_iter(
    cx: &LateContext<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
    parent_expr_span: Span,
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
        && SpanlessEq::new(cx).eq_expr(left_expr, struct_expr)
        && let hir::ExprKind::MethodCall(_, _, [closure_expr], _) = filter_expr.kind
        && let hir::ExprKind::Closure(closure) = closure_expr.kind
        && let filter_body = cx.tcx.hir().body(closure.body)
        && let [filter_params] = filter_body.params
    {
        match filter_params.pat.kind {
            // hir::PatKind::Binding(_, _, _, None) => {
            //     // Be conservative now. Do nothing here.
            //     // TODO: Ideally, we can rewrite the lambda by stripping one level of reference
            // },
            hir::PatKind::Tuple([_, _], _) => {
                // the `&&` reference for the `filter` method will be auto derefed to `ref`
                // so, we can directly use the lambda
                // https://doc.rust-lang.org/reference/patterns.html#binding-modes
                make_span_lint_and_sugg(
                    cx,
                    parent_expr_span,
                    format!(
                        "{}.retain({})",
                        snippet(cx, left_expr.span, ".."),
                        snippet(cx, closure_expr.span, "..")
                    ),
                );
            },
            hir::PatKind::Ref(pat, _) => make_span_lint_and_sugg(
                cx,
                parent_expr_span,
                format!(
                    "{}.retain(|{}| {})",
                    snippet(cx, left_expr.span, ".."),
                    snippet(cx, pat.span, ".."),
                    snippet(cx, filter_body.value.span, "..")
                ),
            ),
            _ => {},
        }
    }
}

fn check_to_owned(
    cx: &LateContext<'_>,
    left_expr: &hir::Expr<'_>,
    target_expr: &hir::Expr<'_>,
    parent_expr_span: Span,
    msrv: &Msrv,
) {
    if msrv.meets(msrvs::STRING_RETAIN)
        && let hir::ExprKind::MethodCall(_, filter_expr, [], _) = &target_expr.kind
        && let Some(to_owned_def_id) = cx.typeck_results().type_dependent_def_id(target_expr.hir_id)
        && cx.tcx.is_diagnostic_item(sym::to_owned_method, to_owned_def_id)
        && let hir::ExprKind::MethodCall(_, chars_expr, [_], _) = &filter_expr.kind
        && let Some(filter_def_id) = cx.typeck_results().type_dependent_def_id(filter_expr.hir_id)
        && match_def_path(cx, filter_def_id, &paths::CORE_ITER_FILTER)
        && let hir::ExprKind::MethodCall(_, str_expr, [], _) = &chars_expr.kind
        && let Some(chars_expr_def_id) = cx.typeck_results().type_dependent_def_id(chars_expr.hir_id)
        && match_def_path(cx, chars_expr_def_id, &paths::STR_CHARS)
        && let ty = cx.typeck_results().expr_ty(str_expr).peel_refs()
        && is_type_lang_item(cx, ty, hir::LangItem::String)
        && SpanlessEq::new(cx).eq_expr(left_expr, str_expr)
        && let hir::ExprKind::MethodCall(_, _, [closure_expr], _) = filter_expr.kind
        && let hir::ExprKind::Closure(closure) = closure_expr.kind
        && let filter_body = cx.tcx.hir().body(closure.body)
        && let [filter_params] = filter_body.params
    {
        if let hir::PatKind::Ref(pat, _) = filter_params.pat.kind {
            make_span_lint_and_sugg(
                cx,
                parent_expr_span,
                format!(
                    "{}.retain(|{}| {})",
                    snippet(cx, left_expr.span, ".."),
                    snippet(cx, pat.span, ".."),
                    snippet(cx, filter_body.value.span, "..")
                ),
            );
        }
        // Be conservative now. Do nothing for the `Binding` case.
        // TODO: Ideally, we can rewrite the lambda by stripping one level of reference
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

fn match_map_type(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr).peel_refs();
    MAP_TYPES.iter().any(|ty| is_type_diagnostic_item(cx, expr_ty, *ty))
}

fn make_span_lint_and_sugg(cx: &LateContext<'_>, span: Span, sugg: String) {
    span_lint_and_sugg(
        cx,
        MANUAL_RETAIN,
        span,
        "this expression can be written more simply using `.retain()`",
        "consider calling `.retain()` instead",
        sugg,
        Applicability::MachineApplicable,
    );
}
