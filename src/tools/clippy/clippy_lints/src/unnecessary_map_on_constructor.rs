use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::get_type_diagnostic_name;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Suggests removing the use of a `map()` (or `map_err()`) method when an `Option` or `Result`
    /// is being constructed.
    ///
    /// ### Why is this bad?
    /// It introduces unnecessary complexity. Instead, the function can be called before
    /// constructing the `Option` or `Result` from its return value.
    ///
    /// ### Example
    /// ```no_run
    /// Some(4).map(i32::swap_bytes)
    /// # ;
    /// ```
    /// Use instead:
    /// ```no_run
    /// Some(i32::swap_bytes(4))
    /// # ;
    /// ```
    #[clippy::version = "1.74.0"]
    pub UNNECESSARY_MAP_ON_CONSTRUCTOR,
    complexity,
    "using `map`/`map_err` on `Option` or `Result` constructors"
}
declare_lint_pass!(UnnecessaryMapOnConstructor => [UNNECESSARY_MAP_ON_CONSTRUCTOR]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryMapOnConstructor {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx rustc_hir::Expr<'tcx>) {
        if expr.span.from_expansion() {
            return;
        }
        if let hir::ExprKind::MethodCall(path, recv, [map_arg], ..) = expr.kind
            && let Some(sym::Option | sym::Result) = get_type_diagnostic_name(cx, cx.typeck_results().expr_ty(recv))
        {
            let (constructor_path, constructor_item) = if let hir::ExprKind::Call(constructor, [arg, ..]) = recv.kind
                && let hir::ExprKind::Path(constructor_path) = constructor.kind
            {
                if constructor.span.from_expansion() || arg.span.from_expansion() {
                    return;
                }
                (constructor_path, arg)
            } else {
                return;
            };
            let constructor_symbol = match constructor_path {
                hir::QPath::Resolved(_, path) => {
                    if let Some(path_segment) = path.segments.last() {
                        path_segment.ident.name
                    } else {
                        return;
                    }
                },
                hir::QPath::TypeRelative(_, path) => path.ident.name,
                hir::QPath::LangItem(..) => return,
            };
            match constructor_symbol {
                sym::Some | sym::Ok if path.ident.name == sym::map => (),
                sym::Err if path.ident.name == sym::map_err => (),
                _ => return,
            }

            if let hir::ExprKind::Path(fun) = map_arg.kind {
                if map_arg.span.from_expansion() {
                    return;
                }
                let mut applicability = Applicability::MachineApplicable;
                let fun_snippet = snippet_with_applicability(cx, fun.span(), "_", &mut applicability);
                let constructor_snippet =
                    snippet_with_applicability(cx, constructor_path.span(), "_", &mut applicability);
                let constructor_arg_snippet =
                    snippet_with_applicability(cx, constructor_item.span, "_", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_MAP_ON_CONSTRUCTOR,
                    expr.span,
                    format!(
                        "unnecessary {} on constructor {constructor_snippet}(_)",
                        path.ident.name
                    ),
                    "try",
                    format!("{constructor_snippet}({fun_snippet}({constructor_arg_snippet}))"),
                    applicability,
                );
            }
        }
    }
}
