use if_chain::if_chain;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

use crate::utils::{
    higher::if_block, in_macro_or_desugar, match_type, paths, span_lint_and_then, usage::is_potentially_mutated,
};
use rustc::hir::intravisit::*;
use rustc::hir::*;
use syntax::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for calls of `unwrap[_err]()` that cannot fail.
    ///
    /// **Why is this bad?** Using `if let` or `match` is more idiomatic.
    ///
    /// **Known problems:** Limitations of the borrow checker might make unwrap() necessary sometimes?
    ///
    /// **Example:**
    /// ```rust
    /// if option.is_some() {
    ///     do_something_with(option.unwrap())
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// if let Some(value) = option {
    ///     do_something_with(value)
    /// }
    /// ```
    pub UNNECESSARY_UNWRAP,
    nursery,
    "checks for calls of unwrap[_err]() that cannot fail"
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
    /// if option.is_none() {
    ///     do_something_with(option.unwrap())
    /// }
    /// ```
    ///
    /// This code will always panic. The if condition should probably be inverted.
    pub PANICKING_UNWRAP,
    nursery,
    "checks for calls of unwrap[_err]() that will always fail"
}

/// Visitor that keeps track of which variables are unwrappable.
struct UnwrappableVariablesVisitor<'a, 'tcx> {
    unwrappables: Vec<UnwrapInfo<'tcx>>,
    cx: &'a LateContext<'a, 'tcx>,
}
/// Contains information about whether a variable can be unwrapped.
#[derive(Copy, Clone, Debug)]
struct UnwrapInfo<'tcx> {
    /// The variable that is checked
    ident: &'tcx Path,
    /// The check, like `x.is_ok()`
    check: &'tcx Expr,
    /// Whether `is_some()` or `is_ok()` was called (as opposed to `is_err()` or `is_none()`).
    safe_to_unwrap: bool,
}

/// Collects the information about unwrappable variables from an if condition
/// The `invert` argument tells us whether the condition is negated.
fn collect_unwrap_info<'a, 'tcx>(
    cx: &'a LateContext<'a, 'tcx>,
    expr: &'tcx Expr,
    invert: bool,
) -> Vec<UnwrapInfo<'tcx>> {
    if let ExprKind::Binary(op, left, right) = &expr.node {
        match (invert, op.node) {
            (false, BinOpKind::And) | (false, BinOpKind::BitAnd) | (true, BinOpKind::Or) | (true, BinOpKind::BitOr) => {
                let mut unwrap_info = collect_unwrap_info(cx, left, invert);
                unwrap_info.append(&mut collect_unwrap_info(cx, right, invert));
                return unwrap_info;
            },
            _ => (),
        }
    } else if let ExprKind::Unary(UnNot, expr) = &expr.node {
        return collect_unwrap_info(cx, expr, !invert);
    } else {
        if_chain! {
            if let ExprKind::MethodCall(method_name, _, args) = &expr.node;
            if let ExprKind::Path(QPath::Resolved(None, path)) = &args[0].node;
            let ty = cx.tables.expr_ty(&args[0]);
            if match_type(cx, ty, &paths::OPTION) || match_type(cx, ty, &paths::RESULT);
            let name = method_name.ident.as_str();
            if ["is_some", "is_none", "is_ok", "is_err"].contains(&&*name);
            then {
                assert!(args.len() == 1);
                let unwrappable = match name.as_ref() {
                    "is_some" | "is_ok" => true,
                    "is_err" | "is_none" => false,
                    _ => unreachable!(),
                };
                let safe_to_unwrap = unwrappable != invert;
                return vec![UnwrapInfo { ident: path, check: expr, safe_to_unwrap }];
            }
        }
    }
    Vec::new()
}

impl<'a, 'tcx> UnwrappableVariablesVisitor<'a, 'tcx> {
    fn visit_branch(&mut self, cond: &'tcx Expr, branch: &'tcx Expr, else_branch: bool) {
        let prev_len = self.unwrappables.len();
        for unwrap_info in collect_unwrap_info(self.cx, cond, else_branch) {
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
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if let Some((cond, then, els)) = if_block(&expr) {
            walk_expr(self, cond);
            self.visit_branch(cond, then, false);
            if let Some(els) = els {
                self.visit_branch(cond, els, true);
            }
        } else {
            // find `unwrap[_err]()` calls:
            if_chain! {
                if let ExprKind::MethodCall(ref method_name, _, ref args) = expr.node;
                if let ExprKind::Path(QPath::Resolved(None, ref path)) = args[0].node;
                if [sym!(unwrap), sym!(unwrap_err)].contains(&method_name.ident.name);
                let call_to_unwrap = method_name.ident.name == sym!(unwrap);
                if let Some(unwrappable) = self.unwrappables.iter()
                    .find(|u| u.ident.res == path.res);
                then {
                    if call_to_unwrap == unwrappable.safe_to_unwrap {
                        span_lint_and_then(
                            self.cx,
                            UNNECESSARY_UNWRAP,
                            expr.span,
                            &format!("You checked before that `{}()` cannot fail. \
                            Instead of checking and unwrapping, it's better to use `if let` or `match`.",
                            method_name.ident.name),
                            |db| { db.span_label(unwrappable.check.span, "the check is happening here"); },
                        );
                    } else {
                        span_lint_and_then(
                            self.cx,
                            PANICKING_UNWRAP,
                            expr.span,
                            &format!("This call to `{}()` will always panic.",
                            method_name.ident.name),
                            |db| { db.span_label(unwrappable.check.span, "because of this check"); },
                        );
                    }
                }
            }
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.cx.tcx.hir())
    }
}

declare_lint_pass!(Unwrap => [PANICKING_UNWRAP, UNNECESSARY_UNWRAP]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Unwrap {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        fn_id: HirId,
    ) {
        if in_macro_or_desugar(span) {
            return;
        }

        let mut v = UnwrappableVariablesVisitor {
            cx,
            unwrappables: Vec::new(),
        };

        walk_fn(&mut v, kind, decl, body.id(), span, fn_id);
    }
}
