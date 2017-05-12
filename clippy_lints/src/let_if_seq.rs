use rustc::lint::*;
use rustc::hir;
use syntax_pos::{Span, NO_EXPANSION};
use utils::{snippet, span_lint_and_then};

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
declare_lint! {
    pub USELESS_LET_IF_SEQ,
    Warn,
    "unidiomatic `let mut` declaration followed by initialization in `if`"
}

#[derive(Copy,Clone)]
pub struct LetIfSeq;

impl LintPass for LetIfSeq {
    fn get_lints(&self) -> LintArray {
        lint_array!(USELESS_LET_IF_SEQ)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LetIfSeq {
    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx hir::Block) {
        let mut it = block.stmts.iter().peekable();
        while let Some(stmt) = it.next() {
            if_let_chain! {[
                let Some(expr) = it.peek(),
                let hir::StmtDecl(ref decl, _) = stmt.node,
                let hir::DeclLocal(ref decl) = decl.node,
                let hir::PatKind::Binding(mode, def_id, ref name, None) = decl.pat.node,
                let hir::StmtExpr(ref if_, _) = expr.node,
                let hir::ExprIf(ref cond, ref then, ref else_) = if_.node,
                !used_in_expr(cx, def_id, cond),
                let hir::ExprBlock(ref then) = then.node,
                let Some(value) = check_assign(cx, def_id, &*then),
                !used_in_expr(cx, def_id, value),
            ], {
                let span = Span { lo: stmt.span.lo, hi: if_.span.hi, ctxt: NO_EXPANSION };

                let (default_multi_stmts, default) = if let Some(ref else_) = *else_ {
                    if let hir::ExprBlock(ref else_) = else_.node {
                        if let Some(default) = check_assign(cx, def_id, else_) {
                            (else_.stmts.len() > 1, default)
                        } else if let Some(ref default) = decl.init {
                            (true, &**default)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else if let Some(ref default) = decl.init {
                    (false, &**default)
                } else {
                    continue;
                };

                let mutability = match mode {
                    hir::BindByRef(hir::MutMutable) | hir::BindByValue(hir::MutMutable) => "<mut> ",
                    _ => "",
                };

                // FIXME: this should not suggest `mut` if we can detect that the variable is not
                // use mutably after the `if`

                let sug = format!(
                    "let {mut}{name} = if {cond} {{{then} {value} }} else {{{else} {default} }};",
                    mut=mutability,
                    name=name.node,
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
                                   |db| {
                                       db.span_suggestion(span,
                                                          "it is more idiomatic to write",
                                                          sug);
                                       if !mutability.is_empty() {
                                           db.note("you might not need `mut` at all");
                                       }
                                   });
            }}
        }
    }
}

struct UsedVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    id: hir::def_id::DefId,
    used: bool,
}

impl<'a, 'tcx> hir::intravisit::Visitor<'tcx> for UsedVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if_let_chain! {[
            let hir::ExprPath(ref qpath) = expr.node,
            self.id == self.cx.tables.qpath_def(qpath, expr.id).def_id(),
        ], {
            self.used = true;
            return;
        }}
        hir::intravisit::walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> hir::intravisit::NestedVisitorMap<'this, 'tcx> {
        hir::intravisit::NestedVisitorMap::None
    }
}

fn check_assign<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    decl: hir::def_id::DefId,
    block: &'tcx hir::Block
) -> Option<&'tcx hir::Expr> {
    if_let_chain! {[
        block.expr.is_none(),
        let Some(expr) = block.stmts.iter().last(),
        let hir::StmtSemi(ref expr, _) = expr.node,
        let hir::ExprAssign(ref var, ref value) = expr.node,
        let hir::ExprPath(ref qpath) = var.node,
        decl == cx.tables.qpath_def(qpath, var.id).def_id(),
    ], {
        let mut v = UsedVisitor {
            cx: cx,
            id: decl,
            used: false,
        };

        for s in block.stmts.iter().take(block.stmts.len()-1) {
            hir::intravisit::walk_stmt(&mut v, s);

            if v.used {
                return None;
            }
        }

        return Some(value);
    }}

    None
}

fn used_in_expr<'a, 'tcx: 'a>(cx: &LateContext<'a, 'tcx>, id: hir::def_id::DefId, expr: &'tcx hir::Expr) -> bool {
    let mut v = UsedVisitor {
        cx: cx,
        id: id,
        used: false,
    };
    hir::intravisit::walk_expr(&mut v, expr);
    v.used
}
