use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::usage::is_potentially_local_place;
use clippy_utils::{higher, path_to_local, sym};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr, walk_fn};
use rustc_hir::{BinOpKind, Body, Expr, ExprKind, FnDecl, HirId, Node, UnOp};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceWithHirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls of `unwrap[_err]()` that cannot fail.
    ///
    /// ### Why is this bad?
    /// Using `if let` or `match` is more idiomatic.
    ///
    /// ### Example
    /// ```no_run
    /// # let option = Some(0);
    /// # fn do_something_with(_x: usize) {}
    /// if option.is_some() {
    ///     do_something_with(option.unwrap())
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```no_run
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
    /// ```no_run
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
            UnwrappableKind::Option => "Some(<item>)",
            UnwrappableKind::Result => "Ok(<item>)",
        }
    }

    fn error_variant_pattern(self) -> &'static str {
        match self {
            UnwrappableKind::Option => "None",
            UnwrappableKind::Result => "Err(<item>)",
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
    check_name: Symbol,
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
    fn is_relevant_option_call(cx: &LateContext<'_>, ty: Ty<'_>, method_name: Symbol) -> bool {
        is_type_diagnostic_item(cx, ty, sym::Option) && matches!(method_name, sym::is_none | sym::is_some)
    }

    fn is_relevant_result_call(cx: &LateContext<'_>, ty: Ty<'_>, method_name: Symbol) -> bool {
        is_type_diagnostic_item(cx, ty, sym::Result) && matches!(method_name, sym::is_err | sym::is_ok)
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
    } else if let ExprKind::MethodCall(method_name, receiver, [], _) = &expr.kind
        && let Some(local_id) = path_to_local(receiver)
        && let ty = cx.typeck_results().expr_ty(receiver)
        && let name = method_name.ident.name
        && (is_relevant_option_call(cx, ty, name) || is_relevant_result_call(cx, ty, name))
    {
        let unwrappable = matches!(name, sym::is_some | sym::is_ok);
        let safe_to_unwrap = unwrappable != invert;
        let kind = if is_type_diagnostic_item(cx, ty, sym::Option) {
            UnwrappableKind::Option
        } else {
            UnwrappableKind::Result
        };

        return vec![UnwrapInfo {
            local_id,
            if_expr,
            check: expr,
            check_name: name,
            branch,
            safe_to_unwrap,
            kind,
            is_entire_condition,
        }];
    }
    Vec::new()
}

/// A HIR visitor delegate that checks if a local variable of type `Option` or `Result` is mutated,
/// *except* for if `.as_mut()` is called.
/// The reason for why we allow that one specifically is that `.as_mut()` cannot change
/// the variant, and that is important because this lint relies on the fact that
/// `is_some` + `unwrap` is equivalent to `if let Some(..) = ..`, which it would not be if
/// the option is changed to None between `is_some` and `unwrap`, ditto for `Result`.
/// (And also `.as_mut()` is a somewhat common method that is still worth linting on.)
struct MutationVisitor<'tcx> {
    is_mutated: bool,
    local_id: HirId,
    tcx: TyCtxt<'tcx>,
}

/// Checks if the parent of the expression pointed at by the given `HirId` is a call to
/// `.as_mut()`.
///
/// Used by the mutation visitor to specifically allow `.as_mut()` calls.
/// In particular, the `HirId` that the visitor receives is the id of the local expression
/// (i.e. the `x` in `x.as_mut()`), and that is the reason for why we care about its parent
/// expression: that will be where the actual method call is.
fn is_as_mut_use(tcx: TyCtxt<'_>, expr_id: HirId) -> bool {
    if let Node::Expr(mutating_expr) = tcx.parent_hir_node(expr_id)
        && let ExprKind::MethodCall(path, _, [], _) = mutating_expr.kind
    {
        path.ident.name == sym::as_mut
    } else {
        false
    }
}

impl<'tcx> Delegate<'tcx> for MutationVisitor<'tcx> {
    fn borrow(&mut self, cat: &PlaceWithHirId<'tcx>, diag_expr_id: HirId, bk: ty::BorrowKind) {
        if let ty::BorrowKind::Mutable = bk
            && is_potentially_local_place(self.local_id, &cat.place)
            && !is_as_mut_use(self.tcx, diag_expr_id)
        {
            self.is_mutated = true;
        }
    }

    fn mutate(&mut self, cat: &PlaceWithHirId<'tcx>, _: HirId) {
        if is_potentially_local_place(self.local_id, &cat.place) {
            self.is_mutated = true;
        }
    }

    fn consume(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn use_cloned(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}

impl<'tcx> UnwrappableVariablesVisitor<'_, 'tcx> {
    fn visit_branch(
        &mut self,
        if_expr: &'tcx Expr<'_>,
        cond: &'tcx Expr<'_>,
        branch: &'tcx Expr<'_>,
        else_branch: bool,
    ) {
        let prev_len = self.unwrappables.len();
        for unwrap_info in collect_unwrap_info(self.cx, if_expr, cond, branch, else_branch, true) {
            let mut delegate = MutationVisitor {
                is_mutated: false,
                local_id: unwrap_info.local_id,
                tcx: self.cx.tcx,
            };

            let vis = ExprUseVisitor::for_clippy(self.cx, cond.hir_id.owner.def_id, &mut delegate);
            vis.walk_expr(cond).into_ok();
            vis.walk_expr(branch).into_ok();

            if delegate.is_mutated {
                // if the variable is mutated, we don't know whether it can be unwrapped.
                // it might have been changed to `None` in between `is_some` + `unwrap`.
                continue;
            }
            self.unwrappables.push(unwrap_info);
        }
        walk_expr(self, branch);
        self.unwrappables.truncate(prev_len);
    }
}

enum AsRefKind {
    AsRef,
    AsMut,
}

/// Checks if the expression is a method call to `as_{ref,mut}` and returns the receiver of it.
/// If it isn't, the expression itself is returned.
fn consume_option_as_ref<'tcx>(expr: &'tcx Expr<'tcx>) -> (&'tcx Expr<'tcx>, Option<AsRefKind>) {
    if let ExprKind::MethodCall(path, recv, [], _) = expr.kind {
        match path.ident.name {
            sym::as_ref => (recv, Some(AsRefKind::AsRef)),
            sym::as_mut => (recv, Some(AsRefKind::AsMut)),
            _ => (expr, None),
        }
    } else {
        (expr, None)
    }
}

impl<'tcx> Visitor<'tcx> for UnwrappableVariablesVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        // Shouldn't lint when `expr` is in macro.
        if expr.span.in_external_macro(self.cx.tcx.sess.source_map()) {
            return;
        }
        // Skip checking inside closures since they are visited through `Unwrap::check_fn()` already.
        if matches!(expr.kind, ExprKind::Closure(_)) {
            return;
        }
        if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr) {
            walk_expr(self, cond);
            self.visit_branch(expr, cond, then, false);
            if let Some(else_inner) = r#else {
                self.visit_branch(expr, cond, else_inner, true);
            }
        } else {
            // find `unwrap[_err]()` or `expect("...")` calls:
            if let ExprKind::MethodCall(method_name, self_arg, ..) = expr.kind
                && let (self_arg, as_ref_kind) = consume_option_as_ref(self_arg)
                && let Some(id) = path_to_local(self_arg)
                && matches!(method_name.ident.name, sym::unwrap | sym::expect | sym::unwrap_err)
                && let call_to_unwrap = matches!(method_name.ident.name, sym::unwrap | sym::expect)
                && let Some(unwrappable) = self.unwrappables.iter()
                    .find(|u| u.local_id == id)
                // Span contexts should not differ with the conditional branch
                && let span_ctxt = expr.span.ctxt()
                && unwrappable.branch.span.ctxt() == span_ctxt
                && unwrappable.check.span.ctxt() == span_ctxt
            {
                if call_to_unwrap == unwrappable.safe_to_unwrap {
                    let is_entire_condition = unwrappable.is_entire_condition;
                    let unwrappable_variable_name = self.cx.tcx.hir_name(unwrappable.local_id);
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
                        format!(
                            "called `{}` on `{unwrappable_variable_name}` after checking its variant with `{}`",
                            method_name.ident.name, unwrappable.check_name,
                        ),
                        |diag| {
                            if is_entire_condition {
                                diag.span_suggestion(
                                    unwrappable.check.span.with_lo(unwrappable.if_expr.span.lo()),
                                    "try",
                                    format!(
                                        "if let {suggested_pattern} = {borrow_prefix}{unwrappable_variable_name}",
                                        borrow_prefix = match as_ref_kind {
                                            Some(AsRefKind::AsRef) => "&",
                                            Some(AsRefKind::AsMut) => "&mut ",
                                            None => "",
                                        },
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
                        format!("this call to `{}()` will always panic", method_name.ident.name),
                        |diag| {
                            diag.span_label(unwrappable.check.span, "because of this check");
                        },
                    );
                }
            }
            walk_expr(self, expr);
        }
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
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
        fn_id: LocalDefId,
    ) {
        if span.from_expansion() {
            return;
        }

        let mut v = UnwrappableVariablesVisitor {
            unwrappables: Vec::new(),
            cx,
        };

        walk_fn(&mut v, kind, decl, body.id(), fn_id);
    }
}
