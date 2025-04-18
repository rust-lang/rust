use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::msrvs::Msrv;
use clippy_utils::{is_none_arm, msrvs, peel_hir_expr_refs};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Arm, Expr, ExprKind, LangItem, Pat, PatKind, QPath, is_range_literal};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol, sym};

declare_clippy_lint! {
    /// ### What it does
    /// This detects various manual reimplementations of `Option::as_slice`.
    ///
    /// ### Why is this bad?
    /// Those implementations are both more complex than calling `as_slice`
    /// and unlike that incur a branch, pessimizing performance and leading
    /// to more generated code.
    ///
    /// ### Example
    /// ```no_run
    ///# let opt = Some(1);
    /// _ = opt.as_ref().map_or(&[][..], std::slice::from_ref);
    /// _ = match opt.as_ref() {
    ///     Some(f) => std::slice::from_ref(f),
    ///     None => &[],
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    ///# let opt = Some(1);
    /// _ = opt.as_slice();
    /// _ = opt.as_slice();
    /// ```
    #[clippy::version = "1.86.0"]
    pub MANUAL_OPTION_AS_SLICE,
    complexity,
    "manual `Option::as_slice`"
}

pub struct ManualOptionAsSlice {
    msrv: Msrv,
}

impl ManualOptionAsSlice {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(ManualOptionAsSlice => [MANUAL_OPTION_AS_SLICE]);

impl LateLintPass<'_> for ManualOptionAsSlice {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let span = expr.span;
        if span.from_expansion() {
            return;
        }
        match expr.kind {
            ExprKind::Match(scrutinee, [arm1, arm2], _) => {
                if is_none_arm(cx, arm2) && check_arms(cx, arm2, arm1)
                    || is_none_arm(cx, arm1) && check_arms(cx, arm1, arm2)
                {
                    check_as_ref(cx, scrutinee, span, self.msrv);
                }
            },
            ExprKind::If(cond, then, Some(other)) => {
                if let ExprKind::Let(let_expr) = cond.kind
                    && let Some(binding) = extract_ident_from_some_pat(cx, let_expr.pat)
                    && check_some_body(cx, binding, then)
                    && is_empty_slice(cx, other.peel_blocks())
                {
                    check_as_ref(cx, let_expr.init, span, self.msrv);
                }
            },
            ExprKind::MethodCall(seg, callee, [], _) => {
                if seg.ident.name.as_str() == "unwrap_or_default" {
                    check_map(cx, callee, span, self.msrv);
                }
            },
            ExprKind::MethodCall(seg, callee, [or], _) => match seg.ident.name.as_str() {
                "unwrap_or" => {
                    if is_empty_slice(cx, or) {
                        check_map(cx, callee, span, self.msrv);
                    }
                },
                "unwrap_or_else" => {
                    if returns_empty_slice(cx, or) {
                        check_map(cx, callee, span, self.msrv);
                    }
                },
                _ => {},
            },
            ExprKind::MethodCall(seg, callee, [or_else, map], _) => match seg.ident.name.as_str() {
                "map_or" => {
                    if is_empty_slice(cx, or_else) && is_slice_from_ref(cx, map) {
                        check_as_ref(cx, callee, span, self.msrv);
                    }
                },
                "map_or_else" => {
                    if returns_empty_slice(cx, or_else) && is_slice_from_ref(cx, map) {
                        check_as_ref(cx, callee, span, self.msrv);
                    }
                },
                _ => {},
            },
            _ => {},
        }
    }
}

fn check_map(cx: &LateContext<'_>, map: &Expr<'_>, span: Span, msrv: Msrv) {
    if let ExprKind::MethodCall(seg, callee, [mapping], _) = map.kind
        && seg.ident.name == sym::map
        && is_slice_from_ref(cx, mapping)
    {
        check_as_ref(cx, callee, span, msrv);
    }
}

fn check_as_ref(cx: &LateContext<'_>, expr: &Expr<'_>, span: Span, msrv: Msrv) {
    if let ExprKind::MethodCall(seg, callee, [], _) = expr.kind
        && seg.ident.name == sym::as_ref
        && let ty::Adt(adtdef, ..) = cx.typeck_results().expr_ty(callee).kind()
        && cx.tcx.is_diagnostic_item(sym::Option, adtdef.did())
        && msrv.meets(
            cx,
            if clippy_utils::is_in_const_context(cx) {
                msrvs::CONST_OPTION_AS_SLICE
            } else {
                msrvs::OPTION_AS_SLICE
            },
        )
    {
        if let Some(snippet) = clippy_utils::source::snippet_opt(cx, callee.span) {
            span_lint_and_sugg(
                cx,
                MANUAL_OPTION_AS_SLICE,
                span,
                "use `Option::as_slice`",
                "use",
                format!("{snippet}.as_slice()"),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint(cx, MANUAL_OPTION_AS_SLICE, span, "use `Option_as_slice`");
        }
    }
}

fn extract_ident_from_some_pat(cx: &LateContext<'_>, pat: &Pat<'_>) -> Option<Symbol> {
    if let PatKind::TupleStruct(QPath::Resolved(None, path), [binding], _) = pat.kind
        && let Res::Def(DefKind::Ctor(..), def_id) = path.res
        && let PatKind::Binding(_mode, _hir_id, ident, _inner_pat) = binding.kind
        && clippy_utils::is_lang_item_or_ctor(cx, def_id, LangItem::OptionSome)
    {
        Some(ident.name)
    } else {
        None
    }
}

/// Returns true if `expr` is `std::slice::from_ref(<name>)`. Used in `if let`s.
fn check_some_body(cx: &LateContext<'_>, name: Symbol, expr: &Expr<'_>) -> bool {
    if let ExprKind::Call(slice_from_ref, [arg]) = expr.peel_blocks().kind
        && is_slice_from_ref(cx, slice_from_ref)
        && let ExprKind::Path(QPath::Resolved(None, path)) = arg.kind
        && let [seg] = path.segments
    {
        seg.ident.name == name
    } else {
        false
    }
}

fn check_arms(cx: &LateContext<'_>, none_arm: &Arm<'_>, some_arm: &Arm<'_>) -> bool {
    if none_arm.guard.is_none()
        && some_arm.guard.is_none()
        && is_empty_slice(cx, none_arm.body)
        && let Some(name) = extract_ident_from_some_pat(cx, some_arm.pat)
    {
        check_some_body(cx, name, some_arm.body)
    } else {
        false
    }
}

fn returns_empty_slice(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Path(_) => clippy_utils::is_path_diagnostic_item(cx, expr, sym::default_fn),
        ExprKind::Closure(cl) => is_empty_slice(cx, cx.tcx.hir_body(cl.body).value),
        _ => false,
    }
}

/// Returns if expr returns an empty slice. If:
/// - An indexing operation to an empty array with a built-in range. `[][..]`
/// - An indexing operation with a zero-ended range. `expr[..0]`
/// - A reference to an empty array. `&[]`
/// - Or a call to `Default::default`.
fn is_empty_slice(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr = peel_hir_expr_refs(expr.peel_blocks()).0;
    match expr.kind {
        ExprKind::Index(arr, range, _) => match arr.kind {
            ExprKind::Array([]) => is_range_literal(range),
            ExprKind::Array(_) => {
                let Some(range) = clippy_utils::higher::Range::hir(range) else {
                    return false;
                };
                range.end.is_some_and(|e| clippy_utils::is_integer_const(cx, e, 0))
            },
            _ => false,
        },
        ExprKind::Array([]) => true,
        ExprKind::Call(def, []) => clippy_utils::is_path_diagnostic_item(cx, def, sym::default_fn),
        _ => false,
    }
}

fn is_slice_from_ref(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    clippy_utils::is_expr_path_def_path(cx, expr, &["core", "slice", "raw", "from_ref"])
}
