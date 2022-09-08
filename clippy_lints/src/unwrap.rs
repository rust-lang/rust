use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::higher;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{path_to_local, usage::is_potentially_mutated};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, walk_fn, FnKind, Visitor};
use rustc_hir::{BinOpKind, Body, Expr, ExprKind, FnDecl, HirId, PathSegment, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls of `unwrap[_err]()` that cannot fail.
    ///
    /// ### Why is this bad?
    /// Using `if let` or `match` is more idiomatic.
    ///
    /// ### Example
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
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_UNWRAP,
    complexity,
    "checks for calls of `unwrap[_err]()` that cannot fail"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls of `unwrap[_err]()` that will always fail.
    ///
    /// ### Why is this bad?
    /// If panicking is desired, an explicit `panic!()` should be used.
    ///
    /// ### Known problems
    /// This lint only checks `if` conditions not assignments.
    /// So something like `let x: Option<()> = None; x.unwrap();` will not be recognized.
    ///
    /// ### Example
    /// ```rust
    /// # let option = Some(0);
    /// # fn do_something_with(_x: usize) {}
    /// if option.is_none() {
    ///     do_something_with(option.unwrap())
    /// }
    /// ```
    ///
    /// This code will always panic. The if condition should probably be inverted.
    #[clippy::version = "pre 1.29.0"]
    pub PANICKING_UNWRAP,
    correctness,
    "checks for calls of `unwrap[_err]()` that will always fail"
}

/// Visitor that keeps track of which variables are unwrappable.
struct UnwrappableVariablesVisitor<'a, 'tcx> {
    unwrappables: Vec<UnwrapInfo<'tcx>>,
    cx: &'a LateContext<'tcx>,
}

/// What kind of unwrappable this is.
#[derive(Copy, Clone, Debug)]
enum UnwrappableKind {
    Option,
    Result,
}

impl UnwrappableKind {
    fn success_variant_pattern(self) -> &'static str {
        match self {
            UnwrappableKind::Option => "Some(..)",
            UnwrappableKind::Result => "Ok(..)",
        }
    }

    fn error_variant_pattern(self) -> &'static str {
        match self {
            UnwrappableKind::Option => "None",
            UnwrappableKind::Result => "Err(..)",
        }
    }
}

/// Contains information about whether a variable can be unwrapped.
#[derive(Copy, Clone, Debug)]
struct UnwrapInfo<'tcx> {
    /// The variable that is checked
    local_id: HirId,
    /// The if itself
    if_expr: &'tcx Expr<'tcx>,
    /// The check, like `x.is_ok()`
    check: &'tcx Expr<'tcx>,
    /// The check's name, like `is_ok`
    check_name: &'tcx PathSegment<'tcx>,
    /// The branch where the check takes place, like `if x.is_ok() { .. }`
    branch: &'tcx Expr<'tcx>,
    /// Whether `is_some()` or `is_ok()` was called (as opposed to `is_err()` or `is_none()`).
    safe_to_unwrap: bool,
    /// What kind of unwrappable this is.
    kind: UnwrappableKind,
    /// If the check is the entire condition (`if x.is_ok()`) or only a part of it (`foo() &&
    /// x.is_ok()`)
    is_entire_condition: bool,
}

/// Collects the information about unwrappable variables from an if condition
/// The `invert` argument tells us whether the condition is negated.
fn collect_unwrap_info<'tcx>(
    cx: &LateContext<'tcx>,
    if_expr: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
    branch: &'tcx Expr<'_>,
    invert: bool,
    is_entire_condition: bool,
) -> Vec<UnwrapInfo<'tcx>> {
    fn is_relevant_option_call(cx: &LateContext<'_>, ty: Ty<'_>, method_name: &str) -> bool {
        is_type_diagnostic_item(cx, ty, sym::Option) && ["is_some", "is_none"].contains(&method_name)
    }

    fn is_relevant_result_call(cx: &LateContext<'_>, ty: Ty<'_>, method_name: &str) -> bool {
        is_type_diagnostic_item(cx, ty, sym::Result) && ["is_ok", "is_err"].contains(&method_name)
    }

    if let ExprKind::Binary(op, left, right) = &expr.kind {
        match (invert, op.node) {
            (false, BinOpKind::And | BinOpKind::BitAnd) | (true, BinOpKind::Or | BinOpKind::BitOr) => {
                let mut unwrap_info = collect_unwrap_info(cx, if_expr, left, branch, invert, false);
                unwrap_info.append(&mut collect_unwrap_info(cx, if_expr, right, branch, invert, false));
                return unwrap_info;
            },
            _ => (),
        }
    } else if let ExprKind::Unary(UnOp::Not, expr) = &expr.kind {
        return collect_unwrap_info(cx, if_expr, expr, branch, !invert, false);
    } else {
        if_chain! {
            if let ExprKind::MethodCall(method_name, receiver, args, _) = &expr.kind;
            if let Some(local_id) = path_to_local(receiver);
            let ty = cx.typeck_results().expr_ty(receiver);
            let name = method_name.ident.as_str();
            if is_relevant_option_call(cx, ty, name) || is_relevant_result_call(cx, ty, name);
            then {
                assert!(args.is_empty());
                let unwrappable = match name {
                    "is_some" | "is_ok" => true,
                    "is_err" | "is_none" => false,
                    _ => unreachable!(),
                };
                let safe_to_unwrap = unwrappable != invert;
                let kind = if is_type_diagnostic_item(cx, ty, sym::Option) {
                    UnwrappableKind::Option
                } else {
                    UnwrappableKind::Result
                };

                return vec![
                    UnwrapInfo {
                        local_id,
                        if_expr,
                        check: expr,
                        check_name: method_name,
                        branch,
                        safe_to_unwrap,
                        kind,
                        is_entire_condition,
                    }
                ]
            }
        }
    }
    Vec::new()
}

impl<'a, 'tcx> UnwrappableVariablesVisitor<'a, 'tcx> {
    fn visit_branch(
        &mut self,
        if_expr: &'tcx Expr<'_>,
        cond: &'tcx Expr<'_>,
        branch: &'tcx Expr<'_>,
        else_branch: bool,
    ) {
        let prev_len = self.unwrappables.len();
        for unwrap_info in collect_unwrap_info(self.cx, if_expr, cond, branch, else_branch, true) {
            if is_potentially_mutated(unwrap_info.local_id, cond, self.cx)
                || is_potentially_mutated(unwrap_info.local_id, branch, self.cx)
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
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        // Shouldn't lint when `expr` is in macro.
        if in_external_macro(self.cx.tcx.sess, expr.span) {
            return;
        }
        if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr) {
            walk_expr(self, cond);
            self.visit_branch(expr, cond, then, false);
            if let Some(else_inner) = r#else {
                self.visit_branch(expr, cond, else_inner, true);
            }
        } else {
            // find `unwrap[_err]()` calls:
            if_chain! {
                if let ExprKind::MethodCall(method_name, self_arg, ..) = expr.kind;
                if let Some(id) = path_to_local(self_arg);
                if [sym::unwrap, sym::expect, sym!(unwrap_err)].contains(&method_name.ident.name);
                let call_to_unwrap = [sym::unwrap, sym::expect].contains(&method_name.ident.name);
                if let Some(unwrappable) = self.unwrappables.iter()
                    .find(|u| u.local_id == id);
                // Span contexts should not differ with the conditional branch
                let span_ctxt = expr.span.ctxt();
                if unwrappable.branch.span.ctxt() == span_ctxt;
                if unwrappable.check.span.ctxt() == span_ctxt;
                then {
                    if call_to_unwrap == unwrappable.safe_to_unwrap {
                        let is_entire_condition = unwrappable.is_entire_condition;
                        let unwrappable_variable_name = self.cx.tcx.hir().name(unwrappable.local_id);
                        let suggested_pattern = if call_to_unwrap {
                            unwrappable.kind.success_variant_pattern()
                        } else {
                            unwrappable.kind.error_variant_pattern()
                        };

                        span_lint_hir_and_then(
                            self.cx,
                            UNNECESSARY_UNWRAP,
                            expr.hir_id,
                            expr.span,
                            &format!(
                                "called `{}` on `{}` after checking its variant with `{}`",
                                method_name.ident.name,
                                unwrappable_variable_name,
                                unwrappable.check_name.ident.as_str(),
                            ),
                            |diag| {
                                if is_entire_condition {
                                    diag.span_suggestion(
                                        unwrappable.check.span.with_lo(unwrappable.if_expr.span.lo()),
                                        "try",
                                        format!(
                                            "if let {} = {}",
                                            suggested_pattern,
                                            unwrappable_variable_name,
                                        ),
                                        // We don't track how the unwrapped value is used inside the
                                        // block or suggest deleting the unwrap, so we can't offer a
                                        // fixable solution.
                                        Applicability::Unspecified,
                                    );
                                } else {
                                    diag.span_label(unwrappable.check.span, "the check is happening here");
                                    diag.help("try using `if let` or `match`");
                                }
                            },
                        );
                    } else {
                        span_lint_hir_and_then(
                            self.cx,
                            PANICKING_UNWRAP,
                            expr.hir_id,
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

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
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
