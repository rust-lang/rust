use crate::utils::paths;
use crate::utils::{
    is_copy, is_type_diagnostic_item, match_trait_method, remove_blocks, snippet_with_applicability, span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::Mutability;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Ident;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `map(|x| x.clone())` or
    /// dereferencing closures for `Copy` types, on `Iterator` or `Option`,
    /// and suggests `cloned()` or `copied()` instead
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

declare_lint_pass!(MapClone => [MAP_CLONE]);

impl<'tcx> LateLintPass<'tcx> for MapClone {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &hir::Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if_chain! {
            if let hir::ExprKind::MethodCall(ref method, _, ref args, _) = e.kind;
            if args.len() == 2;
            if method.ident.as_str() == "map";
            let ty = cx.typeck_results().expr_ty(&args[0]);
            if is_type_diagnostic_item(cx, ty, sym::option_type) || match_trait_method(cx, e, &paths::ITERATOR);
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = args[1].kind;
            let closure_body = cx.tcx.hir().body(body_id);
            let closure_expr = remove_blocks(&closure_body.value);
            then {
                match closure_body.params[0].pat.kind {
                    hir::PatKind::Ref(ref inner, hir::Mutability::Not) => if let hir::PatKind::Binding(
                        hir::BindingAnnotation::Unannotated, .., name, None
                    ) = inner.kind {
                        if ident_eq(name, closure_expr) {
                            lint(cx, e.span, args[0].span, true);
                        }
                    },
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, .., name, None) => {
                        match closure_expr.kind {
                            hir::ExprKind::Unary(hir::UnOp::UnDeref, ref inner) => {
                                if ident_eq(name, inner) {
                                    if let ty::Ref(.., Mutability::Not) = cx.typeck_results().expr_ty(inner).kind() {
                                        lint(cx, e.span, args[0].span, true);
                                    }
                                }
                            },
                            hir::ExprKind::MethodCall(ref method, _, [obj], _) => if_chain! {
                                if ident_eq(name, obj) && method.ident.name == sym::clone;
                                if match_trait_method(cx, closure_expr, &paths::CLONE_TRAIT);
                                // no autoderefs
                                if !cx.typeck_results().expr_adjustments(obj).iter()
                                    .any(|a| matches!(a.kind, Adjust::Deref(Some(..))));
                                then {
                                    let obj_ty = cx.typeck_results().expr_ty(obj);
                                    if let ty::Ref(_, ty, mutability) = obj_ty.kind() {
                                        if matches!(mutability, Mutability::Not) {
                                            let copy = is_copy(cx, ty);
                                            lint(cx, e.span, args[0].span, copy);
                                        }
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

fn ident_eq(name: Ident, path: &hir::Expr<'_>) -> bool {
    if let hir::ExprKind::Path(hir::QPath::Resolved(None, ref path)) = path.kind {
        path.segments.len() == 1 && path.segments[0].ident == name
    } else {
        false
    }
}

fn lint_needless_cloning(cx: &LateContext<'_>, root: Span, receiver: Span) {
    span_lint_and_sugg(
        cx,
        MAP_CLONE,
        root.trim_start(receiver).unwrap(),
        "you are needlessly cloning iterator elements",
        "remove the `map` call",
        String::new(),
        Applicability::MachineApplicable,
    )
}

fn lint(cx: &LateContext<'_>, replace: Span, root: Span, copied: bool) {
    let mut applicability = Applicability::MachineApplicable;
    if copied {
        span_lint_and_sugg(
            cx,
            MAP_CLONE,
            replace,
            "you are using an explicit closure for copying elements",
            "consider calling the dedicated `copied` method",
            format!(
                "{}.copied()",
                snippet_with_applicability(cx, root, "..", &mut applicability)
            ),
            applicability,
        )
    } else {
        span_lint_and_sugg(
            cx,
            MAP_CLONE,
            replace,
            "you are using an explicit closure for cloning elements",
            "consider calling the dedicated `cloned` method",
            format!(
                "{}.cloned()",
                snippet_with_applicability(cx, root, "..", &mut applicability)
            ),
            applicability,
        )
    }
}
