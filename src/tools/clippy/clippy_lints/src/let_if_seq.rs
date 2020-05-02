use crate::utils::{higher, qpath_res, snippet, span_lint_and_then};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::intravisit;
use rustc_hir::BindingAnnotation;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for variable declarations immediately followed by a
    /// conditional affectation.
    ///
    /// **Why is this bad?** This is not idiomatic Rust.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    pub USELESS_LET_IF_SEQ,
    style,
    "unidiomatic `let mut` declaration followed by initialization in `if`"
}

declare_lint_pass!(LetIfSeq => [USELESS_LET_IF_SEQ]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LetIfSeq {
    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx hir::Block<'_>) {
        let mut it = block.stmts.iter().peekable();
        while let Some(stmt) = it.next() {
            if_chain! {
                if let Some(expr) = it.peek();
                if let hir::StmtKind::Local(ref local) = stmt.kind;
                if let hir::PatKind::Binding(mode, canonical_id, ident, None) = local.pat.kind;
                if let hir::StmtKind::Expr(ref if_) = expr.kind;
                if let Some((ref cond, ref then, ref else_)) = higher::if_block(&if_);
                if !used_in_expr(cx, canonical_id, cond);
                if let hir::ExprKind::Block(ref then, _) = then.kind;
                if let Some(value) = check_assign(cx, canonical_id, &*then);
                if !used_in_expr(cx, canonical_id, value);
                then {
                    let span = stmt.span.to(if_.span);

                    let has_interior_mutability = !cx.tables.node_type(canonical_id).is_freeze(
                        cx.tcx,
                        cx.param_env,
                        span
                    );
                    if has_interior_mutability { return; }

                    let (default_multi_stmts, default) = if let Some(ref else_) = *else_ {
                        if let hir::ExprKind::Block(ref else_, _) = else_.kind {
                            if let Some(default) = check_assign(cx, canonical_id, else_) {
                                (else_.stmts.len() > 1, default)
                            } else if let Some(ref default) = local.init {
                                (true, &**default)
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    } else if let Some(ref default) = local.init {
                        (false, &**default)
                    } else {
                        continue;
                    };

                    let mutability = match mode {
                        BindingAnnotation::RefMut | BindingAnnotation::Mutable => "<mut> ",
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

struct UsedVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    id: hir::HirId,
    used: bool,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for UsedVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            if let hir::ExprKind::Path(ref qpath) = expr.kind;
            if let Res::Local(local_id) = qpath_res(self.cx, qpath, expr.hir_id);
            if self.id == local_id;
            then {
                self.used = true;
                return;
            }
        }
        intravisit::walk_expr(self, expr);
    }
    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::None
    }
}

fn check_assign<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    decl: hir::HirId,
    block: &'tcx hir::Block<'_>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    if_chain! {
        if block.expr.is_none();
        if let Some(expr) = block.stmts.iter().last();
        if let hir::StmtKind::Semi(ref expr) = expr.kind;
        if let hir::ExprKind::Assign(ref var, ref value, _) = expr.kind;
        if let hir::ExprKind::Path(ref qpath) = var.kind;
        if let Res::Local(local_id) = qpath_res(cx, qpath, var.hir_id);
        if decl == local_id;
        then {
            let mut v = UsedVisitor {
                cx,
                id: decl,
                used: false,
            };

            for s in block.stmts.iter().take(block.stmts.len()-1) {
                intravisit::walk_stmt(&mut v, s);

                if v.used {
                    return None;
                }
            }

            return Some(value);
        }
    }

    None
}

fn used_in_expr<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, id: hir::HirId, expr: &'tcx hir::Expr<'_>) -> bool {
    let mut v = UsedVisitor { cx, id, used: false };
    intravisit::walk_expr(&mut v, expr);
    v.used
}
