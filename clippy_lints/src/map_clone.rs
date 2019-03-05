use crate::utils::paths;
use crate::utils::{
    in_macro, match_trait_method, match_type, remove_blocks, snippet_with_applicability, span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc::hir;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;
use syntax::ast::Ident;
use syntax::source_map::Span;

#[derive(Clone)]
pub struct Pass;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `iterator.map(|x| x.clone())` and suggests
    /// `iterator.cloned()` instead
    ///
    /// **Why is this bad?** Readability, this can be written more concisely
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.map(|i| *i);
    /// ```
    ///
    /// The correct use would be:
    ///
    /// ```rust
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.cloned();
    /// ```
    pub MAP_CLONE,
    style,
    "using `iterator.map(|x| x.clone())`, or dereferencing closures for `Copy` types"
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MAP_CLONE)
    }

    fn name(&self) -> &'static str {
        "MapClone"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, e: &hir::Expr) {
        if in_macro(e.span) {
            return;
        }

        if_chain! {
            if let hir::ExprKind::MethodCall(ref method, _, ref args) = e.node;
            if args.len() == 2;
            if method.ident.as_str() == "map";
            let ty = cx.tables.expr_ty(&args[0]);
            if match_type(cx, ty, &paths::OPTION) || match_trait_method(cx, e, &paths::ITERATOR);
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = args[1].node;
            let closure_body = cx.tcx.hir().body(body_id);
            let closure_expr = remove_blocks(&closure_body.value);
            then {
                match closure_body.arguments[0].pat.node {
                    hir::PatKind::Ref(ref inner, _) => if let hir::PatKind::Binding(
                        hir::BindingAnnotation::Unannotated, .., name, None
                    ) = inner.node {
                        if ident_eq(name, closure_expr) {
                            lint(cx, e.span, args[0].span);
                        }
                    },
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, .., name, None) => {
                        match closure_expr.node {
                            hir::ExprKind::Unary(hir::UnOp::UnDeref, ref inner) => {
                                if ident_eq(name, inner) && !cx.tables.expr_ty(inner).is_box() {
                                    lint(cx, e.span, args[0].span);
                                }
                            },
                            hir::ExprKind::MethodCall(ref method, _, ref obj) => {
                                if ident_eq(name, &obj[0]) && method.ident.as_str() == "clone"
                                    && match_trait_method(cx, closure_expr, &paths::CLONE_TRAIT) {

                                    let obj_ty = cx.tables.expr_ty(&obj[0]);
                                    if let ty::Ref(..) = obj_ty.sty {
                                        lint(cx, e.span, args[0].span);
                                    } else {
                                        lint_needless_cloning(cx, e.span, args[0].span);
                                    }
                                }
                            },
                            _ => {},
                        }
                    },
                    _ => {},
                }
            }
        }
    }
}

fn ident_eq(name: Ident, path: &hir::Expr) -> bool {
    if let hir::ExprKind::Path(hir::QPath::Resolved(None, ref path)) = path.node {
        path.segments.len() == 1 && path.segments[0].ident == name
    } else {
        false
    }
}

fn lint_needless_cloning(cx: &LateContext<'_, '_>, root: Span, receiver: Span) {
    span_lint_and_sugg(
        cx,
        MAP_CLONE,
        root.trim_start(receiver).unwrap(),
        "You are needlessly cloning iterator elements",
        "Remove the map call",
        String::new(),
        Applicability::MachineApplicable,
    )
}

fn lint(cx: &LateContext<'_, '_>, replace: Span, root: Span) {
    let mut applicability = Applicability::MachineApplicable;
    span_lint_and_sugg(
        cx,
        MAP_CLONE,
        replace,
        "You are using an explicit closure for cloning elements",
        "Consider calling the dedicated `cloned` method",
        format!(
            "{}.cloned()",
            snippet_with_applicability(cx, root, "..", &mut applicability)
        ),
        applicability,
    )
}
