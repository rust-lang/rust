use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::{get_parent_expr, path_to_local, path_to_local_id};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, HirId, LetStmt, Node, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a read and a write to the same variable where
    /// whether the read occurs before or after the write depends on the evaluation
    /// order of sub-expressions.
    ///
    /// ### Why restrict this?
    /// While [the evaluation order of sub-expressions] is fully specified in Rust,
    /// it still may be confusing to read an expression where the evaluation order
    /// affects its behavior.
    ///
    /// ### Known problems
    /// Code which intentionally depends on the evaluation
    /// order, or which is correct for any evaluation order.
    ///
    /// ### Example
    /// ```no_run
    /// let mut x = 0;
    ///
    /// let a = {
    ///     x = 1;
    ///     1
    /// } + x;
    /// // Unclear whether a is 1 or 2.
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let mut x = 0;
    /// let tmp = {
    ///     x = 1;
    ///     1
    /// };
    /// let a = tmp + x;
    /// ```
    ///
    /// [order]: (https://doc.rust-lang.org/reference/expressions.html?highlight=subexpression#evaluation-order-of-operands)
    #[clippy::version = "pre 1.29.0"]
    pub MIXED_READ_WRITE_IN_EXPRESSION,
    restriction,
    "whether a variable read occurs before a write depends on sub-expression evaluation order"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for diverging calls that are not match arms or
    /// statements.
    ///
    /// ### Why is this bad?
    /// It is often confusing to read. In addition, the
    /// sub-expression evaluation order for Rust is not well documented.
    ///
    /// ### Known problems
    /// Someone might want to use `some_bool || panic!()` as a
    /// shorthand.
    ///
    /// ### Example
    /// ```rust,no_run
    /// # fn b() -> bool { true }
    /// # fn c() -> bool { true }
    /// let a = b() || panic!() || c();
    /// // `c()` is dead, `panic!()` is only called if `b()` returns `false`
    /// let x = (a, b, c, panic!());
    /// // can simply be replaced by `panic!()`
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DIVERGING_SUB_EXPRESSION,
    complexity,
    "whether an expression contains a diverging sub expression"
}

declare_lint_pass!(EvalOrderDependence => [MIXED_READ_WRITE_IN_EXPRESSION, DIVERGING_SUB_EXPRESSION]);

impl<'tcx> LateLintPass<'tcx> for EvalOrderDependence {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Find a write to a local variable.
        let var = if let ExprKind::Assign(lhs, ..) | ExprKind::AssignOp(_, lhs, _) = expr.kind
            && let Some(var) = path_to_local(lhs)
            && expr.span.desugaring_kind().is_none()
        {
            var
        } else {
            return;
        };
        let mut visitor = ReadVisitor {
            cx,
            var,
            write_expr: expr,
            last_expr: expr,
        };
        check_for_unsequenced_reads(&mut visitor);
    }
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Let(local) => {
                if let LetStmt { init: Some(e), .. } = local {
                    DivergenceVisitor { cx }.visit_expr(e);
                }
            },
            StmtKind::Expr(e) | StmtKind::Semi(e) => DivergenceVisitor { cx }.maybe_walk_expr(e),
            StmtKind::Item(..) => {},
        }
    }
}

struct DivergenceVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> DivergenceVisitor<'_, 'tcx> {
    fn maybe_walk_expr(&mut self, e: &'tcx Expr<'_>) {
        match e.kind {
            ExprKind::Closure(..) | ExprKind::If(..) | ExprKind::Loop(..) => {},
            ExprKind::Match(e, arms, _) => {
                self.visit_expr(e);
                for arm in arms {
                    if let Some(if_expr) = arm.guard {
                        self.visit_expr(if_expr);
                    }
                    // make sure top level arm expressions aren't linted
                    self.maybe_walk_expr(arm.body);
                }
            },
            _ => walk_expr(self, e),
        }
    }

    fn report_diverging_sub_expr(&mut self, e: &Expr<'_>) {
        if let Some(macro_call) = root_macro_call_first_node(self.cx, e) {
            if self.cx.tcx.item_name(macro_call.def_id).as_str() == "todo" {
                return;
            }
        }
        span_lint(self.cx, DIVERGING_SUB_EXPRESSION, e.span, "sub-expression diverges");
    }
}

fn stmt_might_diverge(stmt: &Stmt<'_>) -> bool {
    !matches!(stmt.kind, StmtKind::Item(..))
}

impl<'tcx> Visitor<'tcx> for DivergenceVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        match e.kind {
            // fix #10776
            ExprKind::Block(block, ..) => match (block.stmts, block.expr) {
                (stmts, Some(e)) => {
                    if stmts.iter().all(|stmt| !stmt_might_diverge(stmt)) {
                        self.visit_expr(e);
                    }
                },
                ([first @ .., stmt], None) => {
                    if first.iter().all(|stmt| !stmt_might_diverge(stmt)) {
                        match stmt.kind {
                            StmtKind::Expr(e) | StmtKind::Semi(e) => self.visit_expr(e),
                            _ => {},
                        }
                    }
                },
                _ => {},
            },
            ExprKind::Continue(_) | ExprKind::Break(_, _) | ExprKind::Ret(_) => self.report_diverging_sub_expr(e),
            ExprKind::Call(func, _) => {
                let typ = self.cx.typeck_results().expr_ty(func);
                match typ.kind() {
                    ty::FnDef(..) | ty::FnPtr(..) => {
                        let sig = typ.fn_sig(self.cx.tcx);
                        if self.cx.tcx.instantiate_bound_regions_with_erased(sig).output().kind() == &ty::Never {
                            self.report_diverging_sub_expr(e);
                        }
                    },
                    _ => {},
                }
            },
            ExprKind::MethodCall(..) => {
                let borrowed_table = self.cx.typeck_results();
                if borrowed_table.expr_ty(e).is_never() {
                    self.report_diverging_sub_expr(e);
                }
            },
            _ => {
                // do not lint expressions referencing objects of type `!`, as that required a
                // diverging expression
                // to begin with
            },
        }
        self.maybe_walk_expr(e);
    }
    fn visit_block(&mut self, _: &'tcx Block<'_>) {
        // don't continue over blocks, LateLintPass already does that
    }
}

/// Walks up the AST from the given write expression (`vis.write_expr`) looking
/// for reads to the same variable that are unsequenced relative to the write.
///
/// This means reads for which there is a common ancestor between the read and
/// the write such that
///
/// * evaluating the ancestor necessarily evaluates both the read and the write (for example, `&x`
///   and `|| x = 1` don't necessarily evaluate `x`), and
///
/// * which one is evaluated first depends on the order of sub-expression evaluation. Blocks, `if`s,
///   loops, `match`es, and the short-circuiting logical operators are considered to have a defined
///   evaluation order.
///
/// When such a read is found, the lint is triggered.
fn check_for_unsequenced_reads(vis: &mut ReadVisitor<'_, '_>) {
    let mut cur_id = vis.write_expr.hir_id;
    loop {
        let parent_id = vis.cx.tcx.parent_hir_id(cur_id);
        if parent_id == cur_id {
            break;
        }

        let stop_early = match vis.cx.tcx.hir_node(parent_id) {
            Node::Expr(expr) => check_expr(vis, expr),
            Node::Stmt(stmt) => check_stmt(vis, stmt),
            Node::Item(_) => {
                // We reached the top of the function, stop.
                break;
            },
            _ => StopEarly::KeepGoing,
        };
        match stop_early {
            StopEarly::Stop => break,
            StopEarly::KeepGoing => {},
        }

        cur_id = parent_id;
    }
}

/// Whether to stop early for the loop in `check_for_unsequenced_reads`. (If
/// `check_expr` weren't an independent function, this would be unnecessary and
/// we could just use `break`).
enum StopEarly {
    KeepGoing,
    Stop,
}

fn check_expr<'tcx>(vis: &mut ReadVisitor<'_, 'tcx>, expr: &'tcx Expr<'_>) -> StopEarly {
    if expr.hir_id == vis.last_expr.hir_id {
        return StopEarly::KeepGoing;
    }

    match expr.kind {
        ExprKind::Array(_)
        | ExprKind::Tup(_)
        | ExprKind::MethodCall(..)
        | ExprKind::Call(_, _)
        | ExprKind::Assign(..)
        | ExprKind::Index(..)
        | ExprKind::Repeat(_, _)
        | ExprKind::Struct(_, _, _)
        | ExprKind::AssignOp(_, _, _) => {
            walk_expr(vis, expr);
        },
        ExprKind::Binary(op, _, _) => {
            if op.node == BinOpKind::And || op.node == BinOpKind::Or {
                // x && y and x || y always evaluate x first, so these are
                // strictly sequenced.
            } else {
                walk_expr(vis, expr);
            }
        },
        ExprKind::Closure { .. } => {
            // Either
            //
            // * `var` is defined in the closure body, in which case we've reached the top of the enclosing
            //   function and can stop, or
            //
            // * `var` is captured by the closure, in which case, because evaluating a closure does not evaluate
            //   its body, we don't necessarily have a write, so we need to stop to avoid generating false
            //   positives.
            //
            // This is also the only place we need to stop early (grrr).
            return StopEarly::Stop;
        },
        // All other expressions either have only one child or strictly
        // sequence the evaluation order of their sub-expressions.
        _ => {},
    }

    vis.last_expr = expr;

    StopEarly::KeepGoing
}

fn check_stmt<'tcx>(vis: &mut ReadVisitor<'_, 'tcx>, stmt: &'tcx Stmt<'_>) -> StopEarly {
    match stmt.kind {
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => check_expr(vis, expr),
        // If the declaration is of a local variable, check its initializer
        // expression if it has one. Otherwise, keep going.
        StmtKind::Let(local) => local
            .init
            .as_ref()
            .map_or(StopEarly::KeepGoing, |expr| check_expr(vis, expr)),
        StmtKind::Item(..) => StopEarly::KeepGoing,
    }
}

/// A visitor that looks for reads from a variable.
struct ReadVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    /// The ID of the variable we're looking for.
    var: HirId,
    /// The expressions where the write to the variable occurred (for reporting
    /// in the lint).
    write_expr: &'tcx Expr<'tcx>,
    /// The last (highest in the AST) expression we've checked, so we know not
    /// to recheck it.
    last_expr: &'tcx Expr<'tcx>,
}

impl<'tcx> Visitor<'tcx> for ReadVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if expr.hir_id == self.last_expr.hir_id {
            return;
        }

        if path_to_local_id(expr, self.var) {
            // Check that this is a read, not a write.
            if !is_in_assignment_position(self.cx, expr) {
                span_lint_and_then(
                    self.cx,
                    MIXED_READ_WRITE_IN_EXPRESSION,
                    expr.span,
                    format!("unsequenced read of `{}`", self.cx.tcx.hir_name(self.var)),
                    |diag| {
                        diag.span_note(
                            self.write_expr.span,
                            "whether read occurs before this write depends on evaluation order",
                        );
                    },
                );
            }
        }
        match expr.kind {
            // We're about to descend a closure. Since we don't know when (or
            // if) the closure will be evaluated, any reads in it might not
            // occur here (or ever). Like above, bail to avoid false positives.
            ExprKind::Closure{..} |

            // We want to avoid a false positive when a variable name occurs
            // only to have its address taken, so we stop here. Technically,
            // this misses some weird cases, eg.
            //
            // ```rust
            // let mut x = 0;
            // let a = foo(&{x = 1; x}, x);
            // ```
            //
            // TODO: fix this
            ExprKind::AddrOf(_, _, _) => {
                return;
            }
            _ => {}
        }

        walk_expr(self, expr);
    }
}

/// Returns `true` if `expr` is the LHS of an assignment, like `expr = ...`.
fn is_in_assignment_position(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr) {
        if let ExprKind::Assign(lhs, ..) = parent.kind {
            return lhs.hir_id == expr.hir_id;
        }
    }
    false
}
