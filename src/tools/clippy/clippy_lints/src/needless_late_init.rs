use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::needs_ordered_drop;
use clippy_utils::visitors::{for_each_expr, for_each_expr_without_closures, is_local_used};
use core::ops::ControlFlow;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::{
    BindingMode, Block, Expr, ExprKind, HirId, HirIdMap, HirIdSet, LetStmt, LocalSource, MatchSource, Node, Pat,
    PatKind, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use std::borrow::Cow;

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

impl_lint_pass!(NeedlessLateInit<'_> => [NEEDLESS_LATE_INIT]);

pub struct NeedlessLateInit<'tcx> {
    check_grouped_late_init: bool,
    grouped_late_inits: Vec<(HirId, HirIdMap<GroupedLateInit<'tcx>>)>,
}

impl<'tcx> NeedlessLateInit<'tcx> {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            check_grouped_late_init: conf.check_grouped_late_init,
            grouped_late_inits: Vec::default(),
        }
    }

    fn check_if_or_match(
        &mut self,
        cx: &LateContext<'tcx>,
        local_stmt: &'tcx LetStmt<'tcx>,
        block: &'tcx Block<'tcx>,
        binding_id: HirId,
        usage: Usage<'tcx>,
        exprs: impl IntoIterator<Item = &'tcx Expr<'tcx>>,
    ) {
        let mut assigns: Vec<LocalAssignGroup<'tcx>> = Vec::new();
        for expr in exprs {
            let ty = cx.typeck_results.expr_ty(expr);
            if ty.is_never() {
                continue;
            }
            if !ty.is_unit() {
                return;
            }

            let Some(assign_group) = LocalAssignGroup::new(cx, expr) else {
                return;
            };

            if let Some(last_group) = assigns.last()
                && !assign_group.is_parallel(last_group)
            {
                return;
            }

            assigns.push(assign_group);
        }

        let Some(first_group) = assigns.first() else {
            return;
        };
        // If there are multiple assignments grouped together, lazyly check them after processing the block.
        if first_group.0.len() > 1 {
            if self.check_grouped_late_init {
                let late_inits = if let Some((hir_id, late_inits)) = self.grouped_late_inits.last_mut()
                    && *hir_id == block.hir_id
                {
                    late_inits
                } else {
                    &mut self.grouped_late_inits.push_mut((block.hir_id, HirIdMap::default())).1
                };

                let mut decls = HirIdMap::default();
                decls.insert(binding_id, local_stmt);
                late_inits.insert(usage.expr.hir_id, GroupedLateInit { usage, assigns, decls });
            }

            return;
        }

        if first_group.0[0].lhs_id == binding_id {
            span_lint_and_then(
                cx,
                NEEDLESS_LATE_INIT,
                local_stmt.span,
                "unneeded late initialization",
                |diag| {
                    let mut suggestions = vec![];
                    for group in assigns {
                        suggestions.extend(
                            group
                                .0
                                .iter()
                                .flat_map(|assign| {
                                    let rhs_span = assign.rhs.span.source_callsite();
                                    let mut spans = vec![assign.span.until(rhs_span)];

                                    if rhs_span.hi() != assign.span.hi() {
                                        spans.push(rhs_span.shrink_to_hi().with_hi(assign.span.hi()));
                                    }

                                    spans
                                })
                                .map(|span| (span, String::new())),
                        );
                    }

                    suggestions.push((local_stmt.span, String::new()));
                    let mut applicability = Applicability::MachineApplicable;
                    let let_snippet = local_snippet_without_semicolon(cx, local_stmt, &mut applicability);
                    suggestions.push((usage.span.shrink_to_lo(), format!("{let_snippet} = ")));
                    if usage.needs_semi {
                        suggestions.push((usage.span.shrink_to_hi(), ";".to_owned()));
                    }
                    let binding_name = cx.tcx.hir_name(binding_id);
                    let descriptor = if matches!(usage.expr.kind, ExprKind::If(..)) {
                        "branches"
                    } else {
                        "`match` arms"
                    };
                    diag.multipart_suggestion(
                    format!(
                        "move the declaration `{binding_name}` here and remove the assignments from the {descriptor}",
                    ),
                    suggestions,
                    applicability,
                );
                },
            );
        }
    }
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
            !needs_ordered_drop(cx, cx.typeck_results.pat_ty(pat))
        } else {
            true
        }
    })
}

#[derive(Debug)]
struct LocalAssign<'tcx> {
    lhs_id: HirId,
    rhs: &'tcx Expr<'tcx>,
    span: Span,
}

impl<'tcx> LocalAssign<'tcx> {
    fn new(expr: &'tcx Expr<'tcx>, span: Span) -> Option<Self> {
        if expr.span.from_expansion() {
            return None;
        }

        if let ExprKind::Assign(lhs, rhs, _) = expr.kind
            && !lhs.span.from_expansion()
        {
            return Some(Self {
                lhs_id: lhs.res_local_id()?,
                rhs,
                span,
            });
        }

        None
    }
}

#[derive(Debug)]
struct Usage<'tcx> {
    span: Span,
    expr: &'tcx Expr<'tcx>,
    needs_semi: bool,
}

fn first_usage<'tcx>(
    cx: &LateContext<'tcx>,
    binding_id: HirId,
    local_stmt_id: HirId,
    block: &'tcx Block<'tcx>,
) -> Option<Usage<'tcx>> {
    let significant_drop = needs_ordered_drop(cx, cx.typeck_results.node_type(binding_id));

    block
        .stmts
        .iter()
        .skip_while(|stmt| stmt.hir_id != local_stmt_id)
        .skip(1)
        .take_while(|stmt| !significant_drop || !stmt_needs_ordered_drop(cx, stmt))
        .find(|&stmt| is_local_used(cx, stmt, binding_id))
        .and_then(|stmt| match stmt.kind {
            StmtKind::Expr(expr) => Some(Usage {
                span: stmt.span,
                expr,
                needs_semi: true,
            }),
            StmtKind::Semi(expr) => Some(Usage {
                span: stmt.span,
                expr,
                needs_semi: false,
            }),
            _ => None,
        })
        .or_else(|| {
            block
                .expr
                .filter(|expr| is_local_used(cx, *expr, binding_id))
                .map(|expr| Usage {
                    span: expr.span,
                    expr,
                    needs_semi: true,
                })
        })
}

fn local_snippet_without_semicolon<'a>(
    cx: &LateContext<'_>,
    local: &LetStmt<'_>,
    applicability: &mut Applicability,
) -> Cow<'a, str> {
    let span = local.span.with_hi(match local.ty {
        // let <pat>: <ty>;
        // ~~~~~~~~~~~~~~~
        Some(ty) => ty.span.hi(),
        // let <pat>;
        // ~~~~~~~~~
        None => local.pat.span.hi(),
    });

    snippet_with_applicability(cx, span, "..", applicability)
}

#[derive(Debug)]
struct LocalAssignGroup<'tcx>(Vec<LocalAssign<'tcx>>);

impl<'tcx> LocalAssignGroup<'tcx> {
    fn new(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<Self> {
        match expr.kind {
            ExprKind::Block(Block { expr: Some(expr), .. }, _)
                if let Some(assign) = LocalAssign::new(expr, expr.span) =>
            {
                Some(LocalAssignGroup(vec![assign]))
            },
            ExprKind::Block(Block { expr: None, stmts, .. }, _) => {
                let mut assign_group = Vec::new();
                // Avoid cases when the assignee is used or reassigned in the subsequent assignments
                let mut used_locals = HirIdSet::default();
                for stmt in stmts.iter().rev() {
                    if let StmtKind::Semi(expr) | StmtKind::Expr(expr) = stmt.kind
                        && let Some(assign) = LocalAssign::new(expr, stmt.span)
                        && !used_locals.contains(&assign.lhs_id)
                    {
                        used_locals.insert(assign.lhs_id);
                        for_each_expr(cx.tcx, assign.rhs, |e| {
                            if let Some(id) = e.res_local_id() {
                                used_locals.insert(id);
                            }
                            ControlFlow::<()>::Continue(())
                        });
                        assign_group.push(assign);
                        continue;
                    }

                    break;
                }
                if assign_group.is_empty() {
                    None
                } else {
                    Some(LocalAssignGroup(assign_group))
                }
            },
            ExprKind::Assign(..) if let Some(assign) = LocalAssign::new(expr, expr.span) => {
                Some(LocalAssignGroup(vec![assign]))
            },
            _ => None,
        }
    }

    /// Checks if the assignments in `self` and `other` are parallel, i.e. they have the same number
    /// of assignments and the same assignees in the same order.
    fn is_parallel(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && self.0.iter().zip(other.0.iter()).all(|(a, b)| a.lhs_id == b.lhs_id)
    }
}

#[derive(Debug)]
struct GroupedLateInit<'tcx> {
    usage: Usage<'tcx>,
    assigns: Vec<LocalAssignGroup<'tcx>>,
    decls: HirIdMap<&'tcx LetStmt<'tcx>>,
}

impl<'tcx> LateLintPass<'tcx> for NeedlessLateInit<'tcx> {
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
            && let Some(usage) = first_usage(cx, *binding_id, local_stmt.hir_id, block)
        {
            if self.check_grouped_late_init
                && let Some((hir_id, late_inits)) = self.grouped_late_inits.last_mut()
                && *hir_id == block.hir_id
                && let Some(late_init) = late_inits.get_mut(&usage.expr.hir_id)
            {
                late_init.decls.insert(*binding_id, local);
                return;
            }

            match usage.expr.kind {
                ExprKind::Assign(..)
                    if let Some(assign) = LocalAssign::new(usage.expr, usage.expr.span)
                        && assign.lhs_id == *binding_id =>
                {
                    let mut applicability = Applicability::MachineApplicable;
                    let let_snippet = local_snippet_without_semicolon(cx, local, &mut applicability);
                    let binding_name = cx.tcx.hir_name(*binding_id);
                    let mut msg_span = MultiSpan::from_spans(vec![local_stmt.span, assign.span]);
                    msg_span.push_span_label(local_stmt.span, "created here");
                    msg_span.push_span_label(assign.span, "initialised here");

                    span_lint_and_then(
                        cx,
                        NEEDLESS_LATE_INIT,
                        msg_span,
                        "unneeded late initialization",
                        |diag| {
                            let mut applicability = Applicability::MachineApplicable;
                            let rhs_snippet = snippet_with_applicability(
                                cx,
                                assign.rhs.span.source_callsite(),
                                "..",
                                &mut applicability,
                            );
                            diag.multipart_suggestion(
                                format!("move the declaration `{binding_name}` here"),
                                vec![
                                    (local_stmt.span, String::new()),
                                    (assign.span, format!("{let_snippet} = {rhs_snippet}")),
                                ],
                                applicability,
                            );
                        },
                    );
                },
                ExprKind::If(cond, then_expr, Some(mut else_expr)) if !contains_let(cond) => {
                    // Flatten multiple if branches
                    let mut exprs = vec![then_expr];
                    while let ExprKind::If(cond, then, Some(else_)) = else_expr.kind {
                        if contains_let(cond) {
                            return;
                        }
                        exprs.push(then);
                        else_expr = else_;
                    }
                    exprs.push(else_expr);
                    self.check_if_or_match(cx, local, block, *binding_id, usage, exprs);
                },
                ExprKind::Match(_, arms, MatchSource::Normal) => {
                    self.check_if_or_match(cx, local, block, *binding_id, usage, arms.iter().map(|arm| arm.body));
                },
                _ => {},
            }
        }
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if self.check_grouped_late_init
            && let Some((_, late_inits)) = self.grouped_late_inits.pop_if(|(hir_id, _)| *hir_id == block.hir_id)
        {
            'outer: for (_, late_init) in late_inits {
                if late_init.decls.len() < late_init.assigns[0].0.len() {
                    continue;
                }

                let mut suggestions = vec![];
                for assign in &late_init.assigns[0].0 {
                    if let Some(local) = late_init.decls.get(&assign.lhs_id)
                    // If the local has a type annotation, skip it since removing the annotation might cause type
                    // inference issues while annotating the tuple makes the suggestion harder to read.
                    && local.ty.is_none()
                    {
                        suggestions.push((local.span, String::new()));
                    } else {
                        continue 'outer;
                    }
                }

                span_lint_and_then(
                    cx,
                    NEEDLESS_LATE_INIT,
                    late_init.usage.span,
                    "unneeded late initialization",
                    |diag| {
                        let mut applicability = Applicability::MachineApplicable;
                        for group in &late_init.assigns {
                            let rhs_snippet = group
                                .0
                                .iter()
                                .rev()
                                .map(|assign| {
                                    snippet_with_applicability(
                                        cx,
                                        assign.rhs.span.source_callsite(),
                                        "..",
                                        &mut applicability,
                                    )
                                })
                                .intersperse(", ".into())
                                .collect::<String>();
                            suggestions.push((
                                group.0.last().unwrap().span.to(group.0[0].span),
                                format!("({rhs_snippet})"),
                            ));
                        }
                        let let_snippet = late_init.assigns[0]
                            .0
                            .iter()
                            .rev()
                            .map(|assign| cx.tcx.hir_name(assign.lhs_id).to_string())
                            .intersperse(", ".to_owned())
                            .collect::<String>();
                        suggestions.push((late_init.usage.span.shrink_to_lo(), format!("let ({let_snippet}) = ")));
                        if late_init.usage.needs_semi {
                            suggestions.push((late_init.usage.span.shrink_to_hi(), ";".to_owned()));
                        }
                        let descriptor = if matches!(late_init.usage.expr.kind, ExprKind::If(..)) {
                            "branches"
                        } else {
                            "`match` arms"
                        };
                        diag.multipart_suggestion(
                            format!("move the declarations here and remove the assignments from the {descriptor}"),
                            suggestions,
                            applicability,
                        );
                    },
                );
            }
        }
    }
}
