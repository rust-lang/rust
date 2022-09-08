use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{path_to_local_id, visitors::is_local_used};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{BindingAnnotation, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for variable declarations immediately followed by a
    /// conditional affectation.
    ///
    /// ### Why is this bad?
    /// This is not idiomatic Rust.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let foo;
    ///
    /// if bar() {
    ///     foo = 42;
    /// } else {
    ///     foo = 0;
    /// }
    ///
    /// let mut baz = None;
    ///
    /// if bar() {
    ///     baz = Some(42);
    /// }
    /// ```
    ///
    /// should be written
    ///
    /// ```rust,ignore
    /// let foo = if bar() {
    ///     42
    /// } else {
    ///     0
    /// };
    ///
    /// let baz = if bar() {
    ///     Some(42)
    /// } else {
    ///     None
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_LET_IF_SEQ,
    nursery,
    "unidiomatic `let mut` declaration followed by initialization in `if`"
}

declare_lint_pass!(LetIfSeq => [USELESS_LET_IF_SEQ]);

impl<'tcx> LateLintPass<'tcx> for LetIfSeq {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        let mut it = block.stmts.iter().peekable();
        while let Some(stmt) = it.next() {
            if_chain! {
                if let Some(expr) = it.peek();
                if let hir::StmtKind::Local(local) = stmt.kind;
                if let hir::PatKind::Binding(mode, canonical_id, ident, None) = local.pat.kind;
                if let hir::StmtKind::Expr(if_) = expr.kind;
                if let hir::ExprKind::If(hir::Expr { kind: hir::ExprKind::DropTemps(cond), ..}, then, else_) = if_.kind;
                if !is_local_used(cx, *cond, canonical_id);
                if let hir::ExprKind::Block(then, _) = then.kind;
                if let Some(value) = check_assign(cx, canonical_id, then);
                if !is_local_used(cx, value, canonical_id);
                then {
                    let span = stmt.span.to(if_.span);

                    let has_interior_mutability = !cx.typeck_results().node_type(canonical_id).is_freeze(
                        cx.tcx.at(span),
                        cx.param_env,
                    );
                    if has_interior_mutability { return; }

                    let (default_multi_stmts, default) = if let Some(else_) = else_ {
                        if let hir::ExprKind::Block(else_, _) = else_.kind {
                            if let Some(default) = check_assign(cx, canonical_id, else_) {
                                (else_.stmts.len() > 1, default)
                            } else if let Some(default) = local.init {
                                (true, default)
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    } else if let Some(default) = local.init {
                        (false, default)
                    } else {
                        continue;
                    };

                    let mutability = match mode {
                        BindingAnnotation(_, Mutability::Mut) => "<mut> ",
                        _ => "",
                    };

                    // FIXME: this should not suggest `mut` if we can detect that the variable is not
                    // use mutably after the `if`

                    let sug = format!(
                        "let {mut}{name} = if {cond} {{{then} {value} }} else {{{else} {default} }};",
                        mut=mutability,
                        name=ident.name,
                        cond=snippet(cx, cond.span, "_"),
                        then=if then.stmts.len() > 1 { " ..;" } else { "" },
                        else=if default_multi_stmts { " ..;" } else { "" },
                        value=snippet(cx, value.span, "<value>"),
                        default=snippet(cx, default.span, "<default>"),
                    );
                    span_lint_and_then(cx,
                                       USELESS_LET_IF_SEQ,
                                       span,
                                       "`if _ { .. } else { .. }` is an expression",
                                       |diag| {
                                           diag.span_suggestion(
                                                span,
                                                "it is more idiomatic to write",
                                                sug,
                                                Applicability::HasPlaceholders,
                                            );
                                           if !mutability.is_empty() {
                                               diag.note("you might not need `mut` at all");
                                           }
                                       });
                }
            }
        }
    }
}

fn check_assign<'tcx>(
    cx: &LateContext<'tcx>,
    decl: hir::HirId,
    block: &'tcx hir::Block<'_>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    if_chain! {
        if block.expr.is_none();
        if let Some(expr) = block.stmts.iter().last();
        if let hir::StmtKind::Semi(expr) = expr.kind;
        if let hir::ExprKind::Assign(var, value, _) = expr.kind;
        if path_to_local_id(var, decl);
        then {
            if block.stmts.iter().take(block.stmts.len()-1).any(|stmt| is_local_used(cx, stmt, decl)) {
                None
            } else {
                Some(value)
            }
        } else {
            None
        }
    }
}
