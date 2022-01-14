use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::path_to_local;
use clippy_utils::source::snippet_opt;
use clippy_utils::visitors::{expr_visitor, is_local_used};
use rustc_errors::Applicability;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{Block, Expr, ExprKind, HirId, Local, LocalSource, MatchSource, Node, Pat, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
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
    /// ```rust
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
    /// ```rust
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
    #[clippy::version = "1.58.0"]
    pub NEEDLESS_LATE_INIT,
    style,
    "late initializations that can be replaced by a `let` statement with an initializer"
}
declare_lint_pass!(NeedlessLateInit => [NEEDLESS_LATE_INIT]);

fn contains_assign_expr<'tcx>(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'tcx>) -> bool {
    let mut seen = false;
    expr_visitor(cx, |expr| {
        if let ExprKind::Assign(..) = expr.kind {
            seen = true;
        }

        !seen
    })
    .visit_stmt(stmt);

    seen
}

#[derive(Debug)]
struct LocalAssign {
    lhs_id: HirId,
    lhs_span: Span,
    rhs_span: Span,
    span: Span,
}

impl LocalAssign {
    fn from_expr(expr: &Expr<'_>, span: Span) -> Option<Self> {
        if let ExprKind::Assign(lhs, rhs, _) = expr.kind {
            if lhs.span.from_expansion() {
                return None;
            }

            Some(Self {
                lhs_id: path_to_local(lhs)?,
                lhs_span: lhs.span,
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
                if_chain! {
                    if let Some((last, other_stmts)) = block.stmts.split_last();
                    if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = last.kind;

                    let assign = Self::from_expr(expr, last.span)?;

                    // avoid visiting if not needed
                    if assign.lhs_id == binding_id;
                    if other_stmts.iter().all(|stmt| !contains_assign_expr(cx, stmt));

                    then {
                        Some(assign)
                    } else {
                        None
                    }
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
        .map(|assignment| Some((assignment.span.until(assignment.rhs_span), String::new())))
        .chain(assignments.iter().map(|assignment| {
            Some((
                assignment.rhs_span.shrink_to_hi().with_hi(assignment.span.hi()),
                String::new(),
            ))
        }))
        .collect::<Option<Vec<(Span, String)>>>()?;

    let applicability = if suggestions.len() > 1 {
        // multiple suggestions don't work with rustfix in multipart_suggest
        // https://github.com/rust-lang/rustfix/issues/141
        Applicability::Unspecified
    } else {
        Applicability::MachineApplicable
    };
    Some((applicability, suggestions))
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
    block
        .stmts
        .iter()
        .skip_while(|stmt| stmt.hir_id != local_stmt_id)
        .skip(1)
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

fn local_snippet_without_semicolon(cx: &LateContext<'_>, local: &Local<'_>) -> Option<String> {
    let span = local.span.with_hi(match local.ty {
        // let <pat>: <ty>;
        // ~~~~~~~~~~~~~~~
        Some(ty) => ty.span.hi(),
        // let <pat>;
        // ~~~~~~~~~
        None => local.pat.span.hi(),
    });

    snippet_opt(cx, span)
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    local: &'tcx Local<'tcx>,
    local_stmt: &'tcx Stmt<'tcx>,
    block: &'tcx Block<'tcx>,
    binding_id: HirId,
) -> Option<()> {
    let usage = first_usage(cx, binding_id, local_stmt.hir_id, block)?;
    let binding_name = cx.tcx.hir().opt_name(binding_id)?;
    let let_snippet = local_snippet_without_semicolon(cx, local)?;

    match usage.expr.kind {
        ExprKind::Assign(..) => {
            let assign = LocalAssign::new(cx, usage.expr, binding_id)?;

            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                local_stmt.span,
                "unneeded late initalization",
                |diag| {
                    diag.tool_only_span_suggestion(
                        local_stmt.span,
                        "remove the local",
                        String::new(),
                        Applicability::MachineApplicable,
                    );

                    diag.span_suggestion(
                        assign.lhs_span,
                        &format!("declare `{}` here", binding_name),
                        let_snippet,
                        Applicability::MachineApplicable,
                    );
                },
            );
        },
        ExprKind::If(_, then_expr, Some(else_expr)) => {
            let (applicability, suggestions) = assignment_suggestions(cx, binding_id, [then_expr, else_expr])?;

            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                local_stmt.span,
                "unneeded late initalization",
                |diag| {
                    diag.tool_only_span_suggestion(local_stmt.span, "remove the local", String::new(), applicability);

                    diag.span_suggestion_verbose(
                        usage.stmt.span.shrink_to_lo(),
                        &format!("declare `{}` here", binding_name),
                        format!("{} = ", let_snippet),
                        applicability,
                    );

                    diag.multipart_suggestion("remove the assignments from the branches", suggestions, applicability);

                    if usage.needs_semi {
                        diag.span_suggestion(
                            usage.stmt.span.shrink_to_hi(),
                            "add a semicolon after the `if` expression",
                            ";".to_string(),
                            applicability,
                        );
                    }
                },
            );
        },
        ExprKind::Match(_, arms, MatchSource::Normal) => {
            let (applicability, suggestions) = assignment_suggestions(cx, binding_id, arms.iter().map(|arm| arm.body))?;

            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                local_stmt.span,
                "unneeded late initalization",
                |diag| {
                    diag.tool_only_span_suggestion(local_stmt.span, "remove the local", String::new(), applicability);

                    diag.span_suggestion_verbose(
                        usage.stmt.span.shrink_to_lo(),
                        &format!("declare `{}` here", binding_name),
                        format!("{} = ", let_snippet),
                        applicability,
                    );

                    diag.multipart_suggestion(
                        "remove the assignments from the `match` arms",
                        suggestions,
                        applicability,
                    );

                    if usage.needs_semi {
                        diag.span_suggestion(
                            usage.stmt.span.shrink_to_hi(),
                            "add a semicolon after the `match` expression",
                            ";".to_string(),
                            applicability,
                        );
                    }
                },
            );
        },
        _ => {},
    };

    Some(())
}

impl<'tcx> LateLintPass<'tcx> for NeedlessLateInit {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        let mut parents = cx.tcx.hir().parent_iter(local.hir_id);

        if_chain! {
            if let Local {
                init: None,
                pat: &Pat {
                    kind: PatKind::Binding(_, binding_id, _, None),
                    ..
                },
                source: LocalSource::Normal,
                ..
            } = local;
            if let Some((_, Node::Stmt(local_stmt))) = parents.next();
            if let Some((_, Node::Block(block))) = parents.next();

            then {
                check(cx, local, local_stmt, block, binding_id);
            }
        }
    }
}
