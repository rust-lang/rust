use crate::utils::{
    differing_macro_contexts, is_type_diagnostic_item, span_lint_and_then, usage::is_potentially_mutated,
};
use if_chain::if_chain;
use rustc_hir::intravisit::{walk_expr, walk_fn, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{BinOpKind, Body, Expr, ExprKind, FnDecl, HirId, Path, QPath, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:** Checks for calls of `unwrap[_err]()` that cannot fail.
    ///
    /// **Why is this bad?** Using `if let` or `match` is more idiomatic.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// # let option = Some(0);
    /// # fn do_something_with(_x: usize) {}
    /// if option.is_some() {
    ///     do_something_with(option.unwrap())
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// # let option = Some(0);
    /// # fn do_something_with(_x: usize) {}
    /// if let Some(value) = option {
    ///     do_something_with(value)
    /// }
    /// ```
    pub UNNECESSARY_UNWRAP,
    complexity,
    "checks for calls of `unwrap[_err]()` that cannot fail"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls of `unwrap[_err]()` that will always fail.
    ///
    /// **Why is this bad?** If panicking is desired, an explicit `panic!()` should be used.
    ///
    /// **Known problems:** This lint only checks `if` conditions not assignments.
    /// So something like `let x: Option<()> = None; x.unwrap();` will not be recognized.
    ///
    /// **Example:**
    /// ```rust
    /// # let option = Some(0);
    /// # fn do_something_with(_x: usize) {}
    /// if option.is_none() {
    ///     do_something_with(option.unwrap())
    /// }
    /// ```
    ///
    /// This code will always panic. The if condition should probably be inverted.
    pub PANICKING_UNWRAP,
    correctness,
    "checks for calls of `unwrap[_err]()` that will always fail"
}

/// Visitor that keeps track of which variables are unwrappable.
struct UnwrappableVariablesVisitor<'a, 'tcx> {
    unwrappables: Vec<UnwrapInfo<'tcx>>,
    cx: &'a LateContext<'tcx>,
}
/// Contains information about whether a variable can be unwrapped.
#[derive(Copy, Clone, Debug)]
struct UnwrapInfo<'tcx> {
    /// The variable that is checked
    ident: &'tcx Path<'tcx>,
    /// The check, like `x.is_ok()`
    check: &'tcx Expr<'tcx>,
    /// The branch where the check takes place, like `if x.is_ok() { .. }`
    branch: &'tcx Expr<'tcx>,
    /// Whether `is_some()` or `is_ok()` was called (as opposed to `is_err()` or `is_none()`).
    safe_to_unwrap: bool,
}

/// Collects the information about unwrappable variables from an if condition
/// The `invert` argument tells us whether the condition is negated.
fn collect_unwrap_info<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    branch: &'tcx Expr<'_>,
    invert: bool,
) -> Vec<UnwrapInfo<'tcx>> {
    fn is_relevant_option_call(cx: &LateContext<'_>, ty: Ty<'_>, method_name: &str) -> bool {
        is_type_diagnostic_item(cx, ty, sym::option_type) && ["is_some", "is_none"].contains(&method_name)
    }

    fn is_relevant_result_call(cx: &LateContext<'_>, ty: Ty<'_>, method_name: &str) -> bool {
        is_type_diagnostic_item(cx, ty, sym::result_type) && ["is_ok", "is_err"].contains(&method_name)
    }

    if let ExprKind::Binary(op, left, right) = &expr.kind {
        match (invert, op.node) {
            (false, BinOpKind::And | BinOpKind::BitAnd) | (true, BinOpKind::Or | BinOpKind::BitOr) => {
                let mut unwrap_info = collect_unwrap_info(cx, left, branch, invert);
                unwrap_info.append(&mut collect_unwrap_info(cx, right, branch, invert));
                return unwrap_info;
            },
            _ => (),
        }
    } else if let ExprKind::Unary(UnOp::UnNot, expr) = &expr.kind {
        return collect_unwrap_info(cx, expr, branch, !invert);
    } else {
        if_chain! {
            if let ExprKind::MethodCall(method_name, _, args, _) = &expr.kind;
            if let ExprKind::Path(QPath::Resolved(None, path)) = &args[0].kind;
            let ty = cx.typeck_results().expr_ty(&args[0]);
            let name = method_name.ident.as_str();
            if is_relevant_option_call(cx, ty, &name) || is_relevant_result_call(cx, ty, &name);
            then {
                assert!(args.len() == 1);
                let unwrappable = match name.as_ref() {
                    "is_some" | "is_ok" => true,
                    "is_err" | "is_none" => false,
                    _ => unreachable!(),
                };
                let safe_to_unwrap = unwrappable != invert;
                return vec![UnwrapInfo { ident: path, check: expr, branch, safe_to_unwrap }];
            }
        }
    }
    Vec::new()
}

impl<'a, 'tcx> UnwrappableVariablesVisitor<'a, 'tcx> {
    fn visit_branch(&mut self, cond: &'tcx Expr<'_>, branch: &'tcx Expr<'_>, else_branch: bool) {
        let prev_len = self.unwrappables.len();
        for unwrap_info in collect_unwrap_info(self.cx, cond, branch, else_branch) {
            if is_potentially_mutated(unwrap_info.ident, cond, self.cx)
                || is_potentially_mutated(unwrap_info.ident, branch, self.cx)
            {
                // if the variable is mutated, we don't know whether it can be unwrapped:
                continue;
            }
            self.unwrappables.push(unwrap_info);
        }
        walk_expr(self, branch);
        self.unwrappables.truncate(prev_len);
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnwrappableVariablesVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        // Shouldn't lint when `expr` is in macro.
        if in_external_macro(self.cx.tcx.sess, expr.span) {
            return;
        }
        if let ExprKind::If(cond, then, els) = &expr.kind {
            walk_expr(self, cond);
            self.visit_branch(cond, then, false);
            if let Some(els) = els {
                self.visit_branch(cond, els, true);
            }
        } else {
            // find `unwrap[_err]()` calls:
            if_chain! {
                if let ExprKind::MethodCall(ref method_name, _, ref args, _) = expr.kind;
                if let ExprKind::Path(QPath::Resolved(None, ref path)) = args[0].kind;
                if [sym::unwrap, sym!(unwrap_err)].contains(&method_name.ident.name);
                let call_to_unwrap = method_name.ident.name == sym::unwrap;
                if let Some(unwrappable) = self.unwrappables.iter()
                    .find(|u| u.ident.res == path.res);
                // Span contexts should not differ with the conditional branch
                if !differing_macro_contexts(unwrappable.branch.span, expr.span);
                if !differing_macro_contexts(unwrappable.branch.span, unwrappable.check.span);
                then {
                    if call_to_unwrap == unwrappable.safe_to_unwrap {
                        span_lint_and_then(
                            self.cx,
                            UNNECESSARY_UNWRAP,
                            expr.span,
                            &format!("you checked before that `{}()` cannot fail, \
                            instead of checking and unwrapping, it's better to use `if let` or `match`",
                            method_name.ident.name),
                            |diag| { diag.span_label(unwrappable.check.span, "the check is happening here"); },
                        );
                    } else {
                        span_lint_and_then(
                            self.cx,
                            PANICKING_UNWRAP,
                            expr.span,
                            &format!("this call to `{}()` will always panic",
                            method_name.ident.name),
                            |diag| { diag.span_label(unwrappable.check.span, "because of this check"); },
                        );
                    }
                }
            }
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}

declare_lint_pass!(Unwrap => [PANICKING_UNWRAP, UNNECESSARY_UNWRAP]);

impl<'tcx> LateLintPass<'tcx> for Unwrap {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        fn_id: HirId,
    ) {
        if span.from_expansion() {
            return;
        }

        let mut v = UnwrappableVariablesVisitor {
            cx,
            unwrappables: Vec::new(),
        };

        walk_fn(&mut v, kind, decl, body.id(), span, fn_id);
    }
}
