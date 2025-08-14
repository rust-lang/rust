use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::HirNode;
use clippy_utils::source::{indent_of, reindent_multiline, snippet, snippet_block_with_context, snippet_with_context};
use clippy_utils::{is_expr_identity_of_pat, is_refutable, peel_blocks};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{Visitor, walk_block, walk_expr, walk_path, walk_stmt};
use rustc_hir::{Arm, Block, Expr, ExprKind, HirId, Node, PatKind, Path, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_span::{Span, Symbol};

use super::MATCH_SINGLE_BINDING;

#[derive(Debug)]
enum AssignmentExpr {
    Assign { span: Span, match_span: Span },
    Local { span: Span, pat_span: Span },
}

#[expect(clippy::too_many_lines)]
pub(crate) fn check<'a>(cx: &LateContext<'a>, ex: &Expr<'a>, arms: &[Arm<'_>], expr: &Expr<'a>) {
    if expr.span.from_expansion() || arms.len() != 1 || is_refutable(cx, arms[0].pat) {
        return;
    }

    let matched_vars = ex.span;
    let bind_names = arms[0].pat.span;
    let match_body = peel_blocks(arms[0].body);
    let mut app = Applicability::MaybeIncorrect;
    let ctxt = expr.span.ctxt();
    let mut snippet_body = snippet_block_with_context(cx, match_body.span, ctxt, "..", Some(expr.span), &mut app).0;

    // Do we need to add ';' to suggestion ?
    if let Node::Stmt(stmt) = cx.tcx.parent_hir_node(expr.hir_id)
        && let StmtKind::Expr(_) = stmt.kind
        && match match_body.kind {
            // We don't need to add a ; to blocks, unless that block is from a macro expansion
            ExprKind::Block(block, _) => block.span.from_expansion(),
            _ => true,
        }
    {
        snippet_body.push(';');
    }

    match arms[0].pat.kind {
        PatKind::Binding(..) | PatKind::Tuple(_, _) | PatKind::Struct(..) => {
            let (target_span, sugg) = match opt_parent_assign_span(cx, ex) {
                Some(AssignmentExpr::Assign { span, match_span }) => {
                    let sugg = sugg_with_curlies(
                        cx,
                        (ex, expr),
                        (bind_names, matched_vars),
                        snippet_body,
                        &mut app,
                        Some(span),
                        true,
                        is_var_binding_used_later(cx, expr, &arms[0]),
                    );

                    span_lint_and_sugg(
                        cx,
                        MATCH_SINGLE_BINDING,
                        span.to(match_span),
                        "this assignment could be simplified",
                        "consider removing the `match` expression",
                        sugg,
                        app,
                    );

                    return;
                },
                Some(AssignmentExpr::Local { span, pat_span }) => (
                    span,
                    format!(
                        "let {} = {};\n{}let {} = {snippet_body};",
                        snippet_with_context(cx, bind_names, ctxt, "..", &mut app).0,
                        snippet_with_context(cx, matched_vars, ctxt, "..", &mut app).0,
                        " ".repeat(indent_of(cx, expr.span).unwrap_or(0)),
                        snippet_with_context(cx, pat_span, ctxt, "..", &mut app).0
                    ),
                ),
                None if is_expr_identity_of_pat(cx, arms[0].pat, ex, false) => {
                    span_lint_and_sugg(
                        cx,
                        MATCH_SINGLE_BINDING,
                        expr.span,
                        "this match could be replaced by its body itself",
                        "consider using the match body instead",
                        snippet_body,
                        Applicability::MachineApplicable,
                    );
                    return;
                },
                None => {
                    let sugg = sugg_with_curlies(
                        cx,
                        (ex, expr),
                        (bind_names, matched_vars),
                        snippet_body,
                        &mut app,
                        None,
                        true,
                        is_var_binding_used_later(cx, expr, &arms[0]),
                    );
                    (expr.span, sugg)
                },
            };

            span_lint_and_sugg(
                cx,
                MATCH_SINGLE_BINDING,
                target_span,
                "this match could be written as a `let` statement",
                "consider using a `let` statement",
                sugg,
                app,
            );
        },
        PatKind::Wild => {
            if ex.can_have_side_effects() {
                let sugg = sugg_with_curlies(
                    cx,
                    (ex, expr),
                    (bind_names, matched_vars),
                    snippet_body,
                    &mut app,
                    None,
                    false,
                    true,
                );

                span_lint_and_sugg(
                    cx,
                    MATCH_SINGLE_BINDING,
                    expr.span,
                    "this match could be replaced by its scrutinee and body",
                    "consider using the scrutinee and body instead",
                    sugg,
                    app,
                );
            } else {
                span_lint_and_sugg(
                    cx,
                    MATCH_SINGLE_BINDING,
                    expr.span,
                    "this match could be replaced by its body itself",
                    "consider using the match body instead",
                    snippet_body,
                    Applicability::MachineApplicable,
                );
            }
        },
        _ => (),
    }
}

struct VarBindingVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    identifiers: FxHashSet<Symbol>,
}

impl<'tcx> Visitor<'tcx> for VarBindingVisitor<'_, 'tcx> {
    type Result = ControlFlow<()>;

    fn visit_path(&mut self, path: &Path<'tcx>, _: HirId) -> Self::Result {
        if let Res::Local(_) = path.res
            && let [segment] = path.segments
            && self.identifiers.contains(&segment.ident.name)
        {
            return ControlFlow::Break(());
        }

        walk_path(self, path)
    }

    fn visit_block(&mut self, block: &'tcx Block<'tcx>) -> Self::Result {
        let before = self.identifiers.clone();
        walk_block(self, block)?;
        self.identifiers = before;
        ControlFlow::Continue(())
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'tcx>) -> Self::Result {
        if let StmtKind::Let(let_stmt) = stmt.kind {
            if let Some(init) = let_stmt.init {
                self.visit_expr(init)?;
            }

            let_stmt.pat.each_binding(|_, _, _, ident| {
                self.identifiers.remove(&ident.name);
            });
        }
        walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> Self::Result {
        match expr.kind {
            ExprKind::If(
                Expr {
                    kind: ExprKind::Let(let_expr),
                    ..
                },
                then,
                else_,
            ) => {
                self.visit_expr(let_expr.init)?;
                let before = self.identifiers.clone();
                let_expr.pat.each_binding(|_, _, _, ident| {
                    self.identifiers.remove(&ident.name);
                });

                self.visit_expr(then)?;
                self.identifiers = before;
                if let Some(else_) = else_ {
                    self.visit_expr(else_)?;
                }
                ControlFlow::Continue(())
            },
            ExprKind::Closure(closure) => {
                let body = self.cx.tcx.hir_body(closure.body);
                let before = self.identifiers.clone();
                for param in body.params {
                    param.pat.each_binding(|_, _, _, ident| {
                        self.identifiers.remove(&ident.name);
                    });
                }
                self.visit_expr(body.value)?;
                self.identifiers = before;
                ControlFlow::Continue(())
            },
            ExprKind::Match(expr, arms, _) => {
                self.visit_expr(expr)?;
                for arm in arms {
                    let before = self.identifiers.clone();
                    arm.pat.each_binding(|_, _, _, ident| {
                        self.identifiers.remove(&ident.name);
                    });
                    if let Some(guard) = arm.guard {
                        self.visit_expr(guard)?;
                    }
                    self.visit_expr(arm.body)?;
                    self.identifiers = before;
                }
                ControlFlow::Continue(())
            },
            _ => walk_expr(self, expr),
        }
    }
}

fn is_var_binding_used_later(cx: &LateContext<'_>, expr: &Expr<'_>, arm: &Arm<'_>) -> bool {
    let Node::Stmt(stmt) = cx.tcx.parent_hir_node(expr.hir_id) else {
        return false;
    };
    let Node::Block(block) = cx.tcx.parent_hir_node(stmt.hir_id) else {
        return false;
    };

    let mut identifiers = FxHashSet::default();
    arm.pat.each_binding(|_, _, _, ident| {
        identifiers.insert(ident.name);
    });

    let mut visitor = VarBindingVisitor { cx, identifiers };
    block
        .stmts
        .iter()
        .skip_while(|s| s.hir_id != stmt.hir_id)
        .skip(1)
        .any(|stmt| matches!(visitor.visit_stmt(stmt), ControlFlow::Break(())))
        || block
            .expr
            .is_some_and(|expr| matches!(visitor.visit_expr(expr), ControlFlow::Break(())))
}

/// Returns true if the `ex` match expression is in a local (`let`) or assign expression
fn opt_parent_assign_span<'a>(cx: &LateContext<'a>, ex: &Expr<'a>) -> Option<AssignmentExpr> {
    if let Node::Expr(parent_arm_expr) = cx.tcx.parent_hir_node(ex.hir_id) {
        return match cx.tcx.parent_hir_node(parent_arm_expr.hir_id) {
            Node::LetStmt(parent_let_expr) => Some(AssignmentExpr::Local {
                span: parent_let_expr.span,
                pat_span: parent_let_expr.pat.span(),
            }),
            Node::Expr(Expr {
                kind: ExprKind::Assign(parent_assign_expr, match_expr, _),
                ..
            }) => Some(AssignmentExpr::Assign {
                span: parent_assign_expr.span,
                match_span: match_expr.span,
            }),
            _ => None,
        };
    }

    None
}

fn expr_in_nested_block(cx: &LateContext<'_>, match_expr: &Expr<'_>) -> bool {
    if let Node::Block(block) = cx.tcx.parent_hir_node(match_expr.hir_id) {
        return block
            .expr
            .map_or_else(|| matches!(block.stmts, [_]), |_| block.stmts.is_empty());
    }
    false
}

fn expr_must_have_curlies(cx: &LateContext<'_>, match_expr: &Expr<'_>) -> bool {
    let parent = cx.tcx.parent_hir_node(match_expr.hir_id);
    if let Node::Expr(Expr {
        kind: ExprKind::Closure(..) | ExprKind::Binary(..),
        ..
    })
    | Node::AnonConst(..) = parent
    {
        return true;
    }

    if let Node::Arm(arm) = &cx.tcx.parent_hir_node(match_expr.hir_id)
        && let ExprKind::Match(..) = arm.body.kind
    {
        return true;
    }

    false
}

fn indent_of_nth_line(snippet: &str, nth: usize) -> Option<usize> {
    snippet
        .lines()
        .nth(nth)
        .and_then(|s| s.find(|c: char| !c.is_whitespace()))
}

fn reindent_snippet_if_in_block(snippet_body: &str, has_assignment: bool) -> String {
    if has_assignment || !snippet_body.starts_with('{') {
        return reindent_multiline(snippet_body, true, indent_of_nth_line(snippet_body, 1));
    }

    let snippet_body = snippet_body.trim_start_matches('{').trim_end_matches('}').trim();
    reindent_multiline(
        snippet_body,
        false,
        indent_of_nth_line(snippet_body, 0).map(|indent| indent.saturating_sub(4)),
    )
}

#[expect(clippy::too_many_arguments)]
fn sugg_with_curlies<'a>(
    cx: &LateContext<'a>,
    (ex, match_expr): (&Expr<'a>, &Expr<'a>),
    (bind_names, matched_vars): (Span, Span),
    mut snippet_body: String,
    applicability: &mut Applicability,
    assignment: Option<Span>,
    needs_var_binding: bool,
    is_var_binding_used_later: bool,
) -> String {
    let assignment_str = assignment.map_or_else(String::new, |span| {
        let mut s = snippet(cx, span, "..").to_string();
        s.push_str(" = ");
        s
    });

    let ctxt = match_expr.span.ctxt();
    let scrutinee = if needs_var_binding {
        format!(
            "let {} = {}",
            snippet_with_context(cx, bind_names, ctxt, "..", applicability).0,
            snippet_with_context(cx, matched_vars, ctxt, "..", applicability).0
        )
    } else {
        snippet_with_context(cx, matched_vars, ctxt, "..", applicability)
            .0
            .to_string()
    };

    let mut indent = " ".repeat(indent_of(cx, ex.span).unwrap_or(0));
    let (mut cbrace_start, mut cbrace_end) = (String::new(), String::new());
    if !expr_in_nested_block(cx, match_expr)
        && ((needs_var_binding && is_var_binding_used_later) || expr_must_have_curlies(cx, match_expr))
    {
        cbrace_end = format!("\n{indent}}}");
        // Fix body indent due to the closure
        indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
        cbrace_start = format!("{{\n{indent}");
        snippet_body = reindent_snippet_if_in_block(&snippet_body, !assignment_str.is_empty());
    }

    format!("{cbrace_start}{scrutinee};\n{indent}{assignment_str}{snippet_body}{cbrace_end}")
}
