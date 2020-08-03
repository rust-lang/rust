use crate::utils::{get_parent_expr, span_lint, span_lint_and_note};
use if_chain::if_chain;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{def, BinOpKind, Block, Expr, ExprKind, Guard, HirId, Local, Node, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for a read and a write to the same variable where
    /// whether the read occurs before or after the write depends on the evaluation
    /// order of sub-expressions.
    ///
    /// **Why is this bad?** It is often confusing to read. In addition, the
    /// sub-expression evaluation order for Rust is not well documented.
    ///
    /// **Known problems:** Code which intentionally depends on the evaluation
    /// order, or which is correct for any evaluation order.
    ///
    /// **Example:**
    /// ```rust
    /// let mut x = 0;
    ///
    /// // Bad
    /// let a = {
    ///     x = 1;
    ///     1
    /// } + x;
    /// // Unclear whether a is 1 or 2.
    ///
    /// // Good
    /// let tmp = {
    ///     x = 1;
    ///     1
    /// };
    /// let a = tmp + x;
    /// ```
    pub EVAL_ORDER_DEPENDENCE,
    complexity,
    "whether a variable read occurs before a write depends on sub-expression evaluation order"
}

declare_clippy_lint! {
    /// **What it does:** Checks for diverging calls that are not match arms or
    /// statements.
    ///
    /// **Why is this bad?** It is often confusing to read. In addition, the
    /// sub-expression evaluation order for Rust is not well documented.
    ///
    /// **Known problems:** Someone might want to use `some_bool || panic!()` as a
    /// shorthand.
    ///
    /// **Example:**
    /// ```rust,no_run
    /// # fn b() -> bool { true }
    /// # fn c() -> bool { true }
    /// let a = b() || panic!() || c();
    /// // `c()` is dead, `panic!()` is only called if `b()` returns `false`
    /// let x = (a, b, c, panic!());
    /// // can simply be replaced by `panic!()`
    /// ```
    pub DIVERGING_SUB_EXPRESSION,
    complexity,
    "whether an expression contains a diverging sub expression"
}

declare_lint_pass!(EvalOrderDependence => [EVAL_ORDER_DEPENDENCE, DIVERGING_SUB_EXPRESSION]);

impl<'tcx> LateLintPass<'tcx> for EvalOrderDependence {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Find a write to a local variable.
        match expr.kind {
            ExprKind::Assign(ref lhs, ..) | ExprKind::AssignOp(_, ref lhs, _) => {
                if let ExprKind::Path(ref qpath) = lhs.kind {
                    if let QPath::Resolved(_, ref path) = *qpath {
                        if path.segments.len() == 1 {
                            if let def::Res::Local(var) = cx.qpath_res(qpath, lhs.hir_id) {
                                let mut visitor = ReadVisitor {
                                    cx,
                                    var,
                                    write_expr: expr,
                                    last_expr: expr,
                                };
                                check_for_unsequenced_reads(&mut visitor);
                            }
                        }
                    }
                }
            },
            _ => {},
        }
    }
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Local(ref local) => {
                if let Local { init: Some(ref e), .. } = **local {
                    DivergenceVisitor { cx }.visit_expr(e);
                }
            },
            StmtKind::Expr(ref e) | StmtKind::Semi(ref e) => DivergenceVisitor { cx }.maybe_walk_expr(e),
            StmtKind::Item(..) => {},
        }
    }
}

struct DivergenceVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> DivergenceVisitor<'a, 'tcx> {
    fn maybe_walk_expr(&mut self, e: &'tcx Expr<'_>) {
        match e.kind {
            ExprKind::Closure(..) => {},
            ExprKind::Match(ref e, arms, _) => {
                self.visit_expr(e);
                for arm in arms {
                    if let Some(Guard::If(if_expr)) = arm.guard {
                        self.visit_expr(if_expr)
                    }
                    // make sure top level arm expressions aren't linted
                    self.maybe_walk_expr(&*arm.body);
                }
            },
            _ => walk_expr(self, e),
        }
    }
    fn report_diverging_sub_expr(&mut self, e: &Expr<'_>) {
        span_lint(self.cx, DIVERGING_SUB_EXPRESSION, e.span, "sub-expression diverges");
    }
}

impl<'a, 'tcx> Visitor<'tcx> for DivergenceVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        match e.kind {
            ExprKind::Continue(_) | ExprKind::Break(_, _) | ExprKind::Ret(_) => self.report_diverging_sub_expr(e),
            ExprKind::Call(ref func, _) => {
                let typ = self.cx.typeck_results().expr_ty(func);
                match typ.kind() {
                    ty::FnDef(..) | ty::FnPtr(_) => {
                        let sig = typ.fn_sig(self.cx.tcx);
                        if let ty::Never = self.cx.tcx.erase_late_bound_regions(&sig).output().kind() {
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
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
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
    let map = &vis.cx.tcx.hir();
    let mut cur_id = vis.write_expr.hir_id;
    loop {
        let parent_id = map.get_parent_node(cur_id);
        if parent_id == cur_id {
            break;
        }
        let parent_node = match map.find(parent_id) {
            Some(parent) => parent,
            None => break,
        };

        let stop_early = match parent_node {
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

fn check_expr<'a, 'tcx>(vis: &mut ReadVisitor<'a, 'tcx>, expr: &'tcx Expr<'_>) -> StopEarly {
    if expr.hir_id == vis.last_expr.hir_id {
        return StopEarly::KeepGoing;
    }

    match expr.kind {
        ExprKind::Array(_)
        | ExprKind::Tup(_)
        | ExprKind::MethodCall(..)
        | ExprKind::Call(_, _)
        | ExprKind::Assign(..)
        | ExprKind::Index(_, _)
        | ExprKind::Repeat(_, _)
        | ExprKind::Struct(_, _, _) => {
            walk_expr(vis, expr);
        },
        ExprKind::Binary(op, _, _) | ExprKind::AssignOp(op, _, _) => {
            if op.node == BinOpKind::And || op.node == BinOpKind::Or {
                // x && y and x || y always evaluate x first, so these are
                // strictly sequenced.
            } else {
                walk_expr(vis, expr);
            }
        },
        ExprKind::Closure(_, _, _, _, _) => {
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

fn check_stmt<'a, 'tcx>(vis: &mut ReadVisitor<'a, 'tcx>, stmt: &'tcx Stmt<'_>) -> StopEarly {
    match stmt.kind {
        StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => check_expr(vis, expr),
        // If the declaration is of a local variable, check its initializer
        // expression if it has one. Otherwise, keep going.
        StmtKind::Local(ref local) => local
            .init
            .as_ref()
            .map_or(StopEarly::KeepGoing, |expr| check_expr(vis, expr)),
        _ => StopEarly::KeepGoing,
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

impl<'a, 'tcx> Visitor<'tcx> for ReadVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if expr.hir_id == self.last_expr.hir_id {
            return;
        }

        match expr.kind {
            ExprKind::Path(ref qpath) => {
                if_chain! {
                    if let QPath::Resolved(None, ref path) = *qpath;
                    if path.segments.len() == 1;
                    if let def::Res::Local(local_id) = self.cx.qpath_res(qpath, expr.hir_id);
                    if local_id == self.var;
                    // Check that this is a read, not a write.
                    if !is_in_assignment_position(self.cx, expr);
                    then {
                        span_lint_and_note(
                            self.cx,
                            EVAL_ORDER_DEPENDENCE,
                            expr.span,
                            "unsequenced read of a variable",
                            Some(self.write_expr.span),
                            "whether read occurs before this write depends on evaluation order"
                        );
                    }
                }
            }
            // We're about to descend a closure. Since we don't know when (or
            // if) the closure will be evaluated, any reads in it might not
            // occur here (or ever). Like above, bail to avoid false positives.
            ExprKind::Closure(_, _, _, _, _) |

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
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Returns `true` if `expr` is the LHS of an assignment, like `expr = ...`.
fn is_in_assignment_position(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr) {
        if let ExprKind::Assign(ref lhs, ..) = parent.kind {
            return lhs.hir_id == expr.hir_id;
        }
    }
    false
}
