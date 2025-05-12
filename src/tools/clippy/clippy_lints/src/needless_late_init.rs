use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::path_to_local;
use clippy_utils::source::{SourceText, SpanRangeExt, snippet};
use clippy_utils::ty::needs_ordered_drop;
use clippy_utils::visitors::{for_each_expr, for_each_expr_without_closures, is_local_used};
use core::ops::ControlFlow;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::{
    BindingMode, Block, Expr, ExprKind, HirId, LetStmt, LocalSource, MatchSource, Node, Pat, PatKind, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for late initializations that can be replaced by a `let` statement
    /// with an initializer.
    ///
    /// ### Why is this bad?
    /// Assigning in the `let` statement is less repetitive.
    ///
    /// ### Example
    /// ```no_run
    /// let a;
    /// a = 1;
    ///
    /// let b;
    /// match 3 {
    ///     0 => b = "zero",
    ///     1 => b = "one",
    ///     _ => b = "many",
    /// }
    ///
    /// let c;
    /// if true {
    ///     c = 1;
    /// } else {
    ///     c = -1;
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let a = 1;
    ///
    /// let b = match 3 {
    ///     0 => "zero",
    ///     1 => "one",
    ///     _ => "many",
    /// };
    ///
    /// let c = if true {
    ///     1
    /// } else {
    ///     -1
    /// };
    /// ```
    #[clippy::version = "1.59.0"]
    pub NEEDLESS_LATE_INIT,
    style,
    "late initializations that can be replaced by a `let` statement with an initializer"
}
declare_lint_pass!(NeedlessLateInit => [NEEDLESS_LATE_INIT]);

fn contains_assign_expr<'tcx>(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'tcx>) -> bool {
    for_each_expr(cx, stmt, |e| {
        if matches!(e.kind, ExprKind::Assign(..)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

fn contains_let(cond: &Expr<'_>) -> bool {
    for_each_expr_without_closures(cond, |e| {
        if matches!(e.kind, ExprKind::Let(_)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

fn stmt_needs_ordered_drop(cx: &LateContext<'_>, stmt: &Stmt<'_>) -> bool {
    let StmtKind::Let(local) = stmt.kind else {
        return false;
    };
    !local.pat.walk_short(|pat| {
        if let PatKind::Binding(.., None) = pat.kind {
            !needs_ordered_drop(cx, cx.typeck_results().pat_ty(pat))
        } else {
            true
        }
    })
}

#[derive(Debug)]
struct LocalAssign {
    lhs_id: HirId,
    rhs_span: Span,
    span: Span,
}

impl LocalAssign {
    fn from_expr(expr: &Expr<'_>, span: Span) -> Option<Self> {
        if expr.span.from_expansion() {
            return None;
        }

        if let ExprKind::Assign(lhs, rhs, _) = expr.kind {
            if lhs.span.from_expansion() {
                return None;
            }

            Some(Self {
                lhs_id: path_to_local(lhs)?,
                rhs_span: rhs.span.source_callsite(),
                span,
            })
        } else {
            None
        }
    }

    fn new<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, binding_id: HirId) -> Option<LocalAssign> {
        let assign = match expr.kind {
            ExprKind::Block(Block { expr: Some(expr), .. }, _) => Self::from_expr(expr, expr.span),
            ExprKind::Block(block, _) => {
                if let Some((last, other_stmts)) = block.stmts.split_last()
                    && let StmtKind::Expr(expr) | StmtKind::Semi(expr) = last.kind

                    && let assign = Self::from_expr(expr, last.span)?

                    // avoid visiting if not needed
                    && assign.lhs_id == binding_id
                    && other_stmts.iter().all(|stmt| !contains_assign_expr(cx, stmt))
                {
                    Some(assign)
                } else {
                    None
                }
            },
            ExprKind::Assign(..) => Self::from_expr(expr, expr.span),
            _ => None,
        }?;

        if assign.lhs_id == binding_id {
            Some(assign)
        } else {
            None
        }
    }
}

fn assignment_suggestions<'tcx>(
    cx: &LateContext<'tcx>,
    binding_id: HirId,
    exprs: impl IntoIterator<Item = &'tcx Expr<'tcx>>,
) -> Option<(Applicability, Vec<(Span, String)>)> {
    let mut assignments = Vec::new();

    for expr in exprs {
        let ty = cx.typeck_results().expr_ty(expr);

        if ty.is_never() {
            continue;
        }
        if !ty.is_unit() {
            return None;
        }

        let assign = LocalAssign::new(cx, expr, binding_id)?;

        assignments.push(assign);
    }

    let suggestions = assignments
        .iter()
        .flat_map(|assignment| {
            let mut spans = vec![assignment.span.until(assignment.rhs_span)];

            if assignment.rhs_span.hi() != assignment.span.hi() {
                spans.push(assignment.rhs_span.shrink_to_hi().with_hi(assignment.span.hi()));
            }

            spans
        })
        .map(|span| (span, String::new()))
        .collect::<Vec<(Span, String)>>();

    match suggestions.len() {
        // All of `exprs` are never types
        // https://github.com/rust-lang/rust-clippy/issues/8911
        0 => None,
        1 => Some((Applicability::MachineApplicable, suggestions)),
        // multiple suggestions don't work with rustfix in multipart_suggest
        // https://github.com/rust-lang/rustfix/issues/141
        _ => Some((Applicability::Unspecified, suggestions)),
    }
}

struct Usage<'tcx> {
    stmt: &'tcx Stmt<'tcx>,
    expr: &'tcx Expr<'tcx>,
    needs_semi: bool,
}

fn first_usage<'tcx>(
    cx: &LateContext<'tcx>,
    binding_id: HirId,
    local_stmt_id: HirId,
    block: &'tcx Block<'tcx>,
) -> Option<Usage<'tcx>> {
    let significant_drop = needs_ordered_drop(cx, cx.typeck_results().node_type(binding_id));

    block
        .stmts
        .iter()
        .skip_while(|stmt| stmt.hir_id != local_stmt_id)
        .skip(1)
        .take_while(|stmt| !significant_drop || !stmt_needs_ordered_drop(cx, stmt))
        .find(|&stmt| is_local_used(cx, stmt, binding_id))
        .and_then(|stmt| match stmt.kind {
            StmtKind::Expr(expr) => Some(Usage {
                stmt,
                expr,
                needs_semi: true,
            }),
            StmtKind::Semi(expr) => Some(Usage {
                stmt,
                expr,
                needs_semi: false,
            }),
            _ => None,
        })
}

fn local_snippet_without_semicolon(cx: &LateContext<'_>, local: &LetStmt<'_>) -> Option<SourceText> {
    let span = local.span.with_hi(match local.ty {
        // let <pat>: <ty>;
        // ~~~~~~~~~~~~~~~
        Some(ty) => ty.span.hi(),
        // let <pat>;
        // ~~~~~~~~~
        None => local.pat.span.hi(),
    });

    span.get_source_text(cx)
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    local: &'tcx LetStmt<'tcx>,
    local_stmt: &'tcx Stmt<'tcx>,
    block: &'tcx Block<'tcx>,
    binding_id: HirId,
) -> Option<()> {
    let usage = first_usage(cx, binding_id, local_stmt.hir_id, block)?;
    let binding_name = cx.tcx.hir_opt_name(binding_id)?;
    let let_snippet = local_snippet_without_semicolon(cx, local)?;

    match usage.expr.kind {
        ExprKind::Assign(..) => {
            let assign = LocalAssign::new(cx, usage.expr, binding_id)?;
            let mut msg_span = MultiSpan::from_spans(vec![local_stmt.span, assign.span]);
            msg_span.push_span_label(local_stmt.span, "created here");
            msg_span.push_span_label(assign.span, "initialised here");

            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                msg_span,
                "unneeded late initialization",
                |diag| {
                    diag.multipart_suggestion(
                        format!("move the declaration `{binding_name}` here"),
                        vec![
                            (local_stmt.span, String::new()),
                            (
                                assign.span,
                                let_snippet.to_owned() + " = " + &snippet(cx, assign.rhs_span, ".."),
                            ),
                        ],
                        Applicability::MachineApplicable,
                    );
                },
            );
        },
        ExprKind::If(cond, then_expr, Some(else_expr)) if !contains_let(cond) => {
            let (applicability, mut suggestions) = assignment_suggestions(cx, binding_id, [then_expr, else_expr])?;

            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                local_stmt.span,
                "unneeded late initialization",
                |diag| {
                    suggestions.push((local_stmt.span, String::new()));
                    suggestions.push((usage.stmt.span.shrink_to_lo(), format!("{let_snippet} = ")));

                    if usage.needs_semi {
                        suggestions.push((usage.stmt.span.shrink_to_hi(), ";".to_owned()));
                    }

                    diag.multipart_suggestion(
                        format!(
                            "move the declaration `{binding_name}` here and remove the assignments from the branches"
                        ),
                        suggestions,
                        applicability,
                    );
                },
            );
        },
        ExprKind::Match(_, arms, MatchSource::Normal) => {
            let (applicability, mut suggestions) =
                assignment_suggestions(cx, binding_id, arms.iter().map(|arm| arm.body))?;

            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                local_stmt.span,
                "unneeded late initialization",
                |diag| {
                    suggestions.push((local_stmt.span, String::new()));
                    suggestions.push((usage.stmt.span.shrink_to_lo(), format!("{let_snippet} = ")));

                    if usage.needs_semi {
                        suggestions.push((usage.stmt.span.shrink_to_hi(), ";".to_owned()));
                    }

                    diag.multipart_suggestion(
                        format!("move the declaration `{binding_name}` here and remove the assignments from the `match` arms"),
                        suggestions,
                        applicability,
                    );
                },
            );
        },
        _ => {},
    }

    Some(())
}

impl<'tcx> LateLintPass<'tcx> for NeedlessLateInit {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx LetStmt<'tcx>) {
        let mut parents = cx.tcx.hir_parent_iter(local.hir_id);
        if let LetStmt {
            init: None,
            pat:
                Pat {
                    kind: PatKind::Binding(BindingMode::NONE, binding_id, _, None),
                    ..
                },
            source: LocalSource::Normal,
            ..
        } = local
            && let Some((_, Node::Stmt(local_stmt))) = parents.next()
            && let Some((_, Node::Block(block))) = parents.next()
        {
            check(cx, local, local_stmt, block, *binding_id);
        }
    }
}
