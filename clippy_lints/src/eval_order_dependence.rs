use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{Visitor, walk_expr, NestedVisitorMap};
use rustc::hir::*;
use rustc::ty;
use rustc::lint::*;
use utils::{get_parent_expr, span_note_and_lint, span_lint};

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
/// let a = {x = 1; 1} + x;
/// // Unclear whether a is 1 or 2.
/// ```
declare_lint! {
    pub EVAL_ORDER_DEPENDENCE,
    Warn,
    "whether a variable read occurs before a write depends on sub-expression evaluation order"
}

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
/// ```rust
/// let a = b() || panic!() || c();
/// // `c()` is dead, `panic!()` is only called if `b()` returns `false`
/// let x = (a, b, c, panic!());
/// // can simply be replaced by `panic!()`
/// ```
declare_lint! {
    pub DIVERGING_SUB_EXPRESSION,
    Warn,
    "whether an expression contains a diverging sub expression"
}

#[derive(Copy, Clone)]
pub struct EvalOrderDependence;

impl LintPass for EvalOrderDependence {
    fn get_lints(&self) -> LintArray {
        lint_array!(EVAL_ORDER_DEPENDENCE, DIVERGING_SUB_EXPRESSION)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EvalOrderDependence {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        // Find a write to a local variable.
        match expr.node {
            ExprAssign(ref lhs, _) |
            ExprAssignOp(_, ref lhs, _) => {
                if let ExprPath(ref qpath) = lhs.node {
                    if let QPath::Resolved(_, ref path) = *qpath {
                        if path.segments.len() == 1 {
                            let var = cx.tables.qpath_def(qpath, lhs.hir_id).def_id();
                            let mut visitor = ReadVisitor {
                                cx: cx,
                                var: var,
                                write_expr: expr,
                                last_expr: expr,
                            };
                            check_for_unsequenced_reads(&mut visitor);
                        }
                    }
                }
            },
            _ => {},
        }
    }
    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt) {
        match stmt.node {
            StmtExpr(ref e, _) |
            StmtSemi(ref e, _) => DivergenceVisitor { cx: cx }.maybe_walk_expr(e),
            StmtDecl(ref d, _) => {
                if let DeclLocal(ref local) = d.node {
                    if let Local { init: Some(ref e), .. } = **local {
                        DivergenceVisitor { cx: cx }.visit_expr(e);
                    }
                }
            },
        }
    }
}

struct DivergenceVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> DivergenceVisitor<'a, 'tcx> {
    fn maybe_walk_expr(&mut self, e: &'tcx Expr) {
        match e.node {
            ExprClosure(.., _) => {},
            ExprMatch(ref e, ref arms, _) => {
                self.visit_expr(e);
                for arm in arms {
                    if let Some(ref guard) = arm.guard {
                        self.visit_expr(guard);
                    }
                    // make sure top level arm expressions aren't linted
                    self.maybe_walk_expr(&*arm.body);
                }
            },
            _ => walk_expr(self, e),
        }
    }
    fn report_diverging_sub_expr(&mut self, e: &Expr) {
        span_lint(self.cx, DIVERGING_SUB_EXPRESSION, e.span, "sub-expression diverges");
    }
}

impl<'a, 'tcx> Visitor<'tcx> for DivergenceVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr) {
        match e.node {
            ExprAgain(_) | ExprBreak(_, _) | ExprRet(_) => self.report_diverging_sub_expr(e),
            ExprCall(ref func, _) => {
                let typ = self.cx.tables.expr_ty(func);
                match typ.sty {
                    ty::TyFnDef(..) | ty::TyFnPtr(_) => {
                        let sig = typ.fn_sig(self.cx.tcx);
                        if let ty::TyNever = self.cx.tcx.erase_late_bound_regions(&sig).output().sty {
                            self.report_diverging_sub_expr(e);
                        }
                    },
                    _ => {},
                }
            },
            ExprMethodCall(..) => {
                let borrowed_table = self.cx.tables;
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
    fn visit_block(&mut self, _: &'tcx Block) {
        // don't continue over blocks, LateLintPass already does that
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

/// Walks up the AST from the given write expression (`vis.write_expr`) looking
/// for reads to the same variable that are unsequenced relative to the write.
///
/// This means reads for which there is a common ancestor between the read and
/// the write such that
///
/// * evaluating the ancestor necessarily evaluates both the read and the write
///   (for example, `&x` and `|| x = 1` don't necessarily evaluate `x`), and
///
/// * which one is evaluated first depends on the order of sub-expression
///   evaluation. Blocks, `if`s, loops, `match`es, and the short-circuiting
///   logical operators are considered to have a defined evaluation order.
///
/// When such a read is found, the lint is triggered.
fn check_for_unsequenced_reads(vis: &mut ReadVisitor) {
    let map = &vis.cx.tcx.hir;
    let mut cur_id = vis.write_expr.id;
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
            map::Node::NodeExpr(expr) => check_expr(vis, expr),
            map::Node::NodeStmt(stmt) => check_stmt(vis, stmt),
            map::Node::NodeItem(_) => {
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

fn check_expr<'a, 'tcx>(vis: &mut ReadVisitor<'a, 'tcx>, expr: &'tcx Expr) -> StopEarly {
    if expr.id == vis.last_expr.id {
        return StopEarly::KeepGoing;
    }

    match expr.node {
        ExprArray(_) |
        ExprTup(_) |
        ExprMethodCall(..) |
        ExprCall(_, _) |
        ExprAssign(_, _) |
        ExprIndex(_, _) |
        ExprRepeat(_, _) |
        ExprStruct(_, _, _) => {
            walk_expr(vis, expr);
        },
        ExprBinary(op, _, _) |
        ExprAssignOp(op, _, _) => {
            if op.node == BiAnd || op.node == BiOr {
                // x && y and x || y always evaluate x first, so these are
                // strictly sequenced.
            } else {
                walk_expr(vis, expr);
            }
        },
        ExprClosure(_, _, _, _, _) => {
            // Either
            //
            // * `var` is defined in the closure body, in which case we've
            //   reached the top of the enclosing function and can stop, or
            //
            // * `var` is captured by the closure, in which case, because
            //   evaluating a closure does not evaluate its body, we don't
            //   necessarily have a write, so we need to stop to avoid
            //   generating false positives.
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

fn check_stmt<'a, 'tcx>(vis: &mut ReadVisitor<'a, 'tcx>, stmt: &'tcx Stmt) -> StopEarly {
    match stmt.node {
        StmtExpr(ref expr, _) |
        StmtSemi(ref expr, _) => check_expr(vis, expr),
        StmtDecl(ref decl, _) => {
            // If the declaration is of a local variable, check its initializer
            // expression if it has one. Otherwise, keep going.
            let local = match decl.node {
                DeclLocal(ref local) => Some(local),
                _ => None,
            };
            local.and_then(|local| local.init.as_ref()).map_or(
                StopEarly::KeepGoing,
                |expr| check_expr(vis, expr),
            )
        },
    }
}

/// A visitor that looks for reads from a variable.
struct ReadVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    /// The id of the variable we're looking for.
    var: DefId,
    /// The expressions where the write to the variable occurred (for reporting
    /// in the lint).
    write_expr: &'tcx Expr,
    /// The last (highest in the AST) expression we've checked, so we know not
    /// to recheck it.
    last_expr: &'tcx Expr,
}

impl<'a, 'tcx> Visitor<'tcx> for ReadVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if expr.id == self.last_expr.id {
            return;
        }

        match expr.node {
            ExprPath(ref qpath) => {
                if let QPath::Resolved(None, ref path) = *qpath {
                    if path.segments.len() == 1 && self.cx.tables.qpath_def(qpath, expr.hir_id).def_id() == self.var {
                        if is_in_assignment_position(self.cx, expr) {
                            // This is a write, not a read.
                        } else {
                            span_note_and_lint(
                                self.cx,
                                EVAL_ORDER_DEPENDENCE,
                                expr.span,
                                "unsequenced read of a variable",
                                self.write_expr.span,
                                "whether read occurs before this write depends on evaluation order"
                            );
                        }
                    }
                }
            }
            // We're about to descend a closure. Since we don't know when (or
            // if) the closure will be evaluated, any reads in it might not
            // occur here (or ever). Like above, bail to avoid false positives.
            ExprClosure(_, _, _, _, _) |

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
            ExprAddrOf(_, _) => {
                return;
            }
            _ => {}
        }

        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

/// Returns true if `expr` is the LHS of an assignment, like `expr = ...`.
fn is_in_assignment_position(cx: &LateContext, expr: &Expr) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr) {
        if let ExprAssign(ref lhs, _) = parent.node {
            return lhs.id == expr.id;
        }
    }
    false
}
