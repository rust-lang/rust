use clippy_utils::diagnostics::{span_lint_hir, span_lint_hir_and_then};
use clippy_utils::higher::{VecInitKind, get_vec_init_kind};
use clippy_utils::source::snippet;
use clippy_utils::{get_enclosing_block, sym};

use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{self as hir, Expr, ExprKind, HirId, LetStmt, PatKind, PathSegment, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// This lint catches reads into a zero-length `Vec`.
    /// Especially in the case of a call to `with_capacity`, this lint warns that read
    /// gets the number of bytes from the `Vec`'s length, not its capacity.
    ///
    /// ### Why is this bad?
    /// Reading zero bytes is almost certainly not the intended behavior.
    ///
    /// ### Known problems
    /// In theory, a very unusual read implementation could assign some semantic meaning
    /// to zero-byte reads. But it seems exceptionally unlikely that code intending to do
    /// a zero-byte read would allocate a `Vec` for it.
    ///
    /// ### Example
    /// ```no_run
    /// use std::io;
    /// fn foo<F: io::Read>(mut f: F) {
    ///     let mut data = Vec::with_capacity(100);
    ///     f.read(&mut data).unwrap();
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::io;
    /// fn foo<F: io::Read>(mut f: F) {
    ///     let mut data = Vec::with_capacity(100);
    ///     data.resize(100, 0);
    ///     f.read(&mut data).unwrap();
    /// }
    /// ```
    #[clippy::version = "1.63.0"]
    pub READ_ZERO_BYTE_VEC,
    nursery,
    "checks for reads into a zero-length `Vec`"
}
declare_lint_pass!(ReadZeroByteVec => [READ_ZERO_BYTE_VEC]);

impl<'tcx> LateLintPass<'tcx> for ReadZeroByteVec {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &hir::Block<'tcx>) {
        for stmt in block.stmts {
            if stmt.span.from_expansion() {
                return;
            }

            if let StmtKind::Let(local) = stmt.kind
                && let LetStmt {
                    pat, init: Some(init), ..
                } = local
                && let PatKind::Binding(_, id, ident, _) = pat.kind
                && let Some(vec_init_kind) = get_vec_init_kind(cx, init)
            {
                let mut visitor = ReadVecVisitor {
                    local_id: id,
                    read_zero_expr: None,
                    has_resize: false,
                };

                let Some(enclosing_block) = get_enclosing_block(cx, id) else {
                    return;
                };
                visitor.visit_block(enclosing_block);

                if let Some(expr) = visitor.read_zero_expr {
                    let applicability = Applicability::MaybeIncorrect;
                    match vec_init_kind {
                        VecInitKind::WithConstCapacity(len) => span_lint_hir_and_then(
                            cx,
                            READ_ZERO_BYTE_VEC,
                            expr.hir_id,
                            expr.span,
                            "reading zero byte data to `Vec`",
                            |diag| {
                                diag.span_suggestion(
                                    expr.span,
                                    "try",
                                    format!("{}.resize({len}, 0); {}", ident, snippet(cx, expr.span, "..")),
                                    applicability,
                                );
                            },
                        ),
                        VecInitKind::WithExprCapacity(hir_id) => {
                            let e = cx.tcx.hir_expect_expr(hir_id);
                            span_lint_hir_and_then(
                                cx,
                                READ_ZERO_BYTE_VEC,
                                expr.hir_id,
                                expr.span,
                                "reading zero byte data to `Vec`",
                                |diag| {
                                    diag.span_suggestion(
                                        expr.span,
                                        "try",
                                        format!(
                                            "{}.resize({}, 0); {}",
                                            ident,
                                            snippet(cx, e.span, ".."),
                                            snippet(cx, expr.span, "..")
                                        ),
                                        applicability,
                                    );
                                },
                            );
                        },
                        _ => {
                            span_lint_hir(
                                cx,
                                READ_ZERO_BYTE_VEC,
                                expr.hir_id,
                                expr.span,
                                "reading zero byte data to `Vec`",
                            );
                        },
                    }
                }
            }
        }
    }
}

struct ReadVecVisitor<'tcx> {
    local_id: HirId,
    read_zero_expr: Option<&'tcx Expr<'tcx>>,
    has_resize: bool,
}

impl<'tcx> Visitor<'tcx> for ReadVecVisitor<'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr<'tcx>) {
        if let ExprKind::MethodCall(path, receiver, args, _) = e.kind {
            let PathSegment { ident, .. } = *path;

            match ident.name {
                sym::read | sym::read_exact => {
                    let [arg] = args else { return };
                    if let ExprKind::AddrOf(_, hir::Mutability::Mut, inner) = arg.kind
                        && let ExprKind::Path(QPath::Resolved(None, inner_path)) = inner.kind
                        && let [inner_seg] = inner_path.segments
                        && let Res::Local(res_id) = inner_seg.res
                        && self.local_id == res_id
                    {
                        self.read_zero_expr = Some(e);
                        return;
                    }
                },
                sym::resize => {
                    // If the Vec is resized, then it's a valid read
                    if let ExprKind::Path(QPath::Resolved(_, inner_path)) = receiver.kind
                        && let Res::Local(res_id) = inner_path.res
                        && self.local_id == res_id
                    {
                        self.has_resize = true;
                        return;
                    }
                },
                _ => {},
            }
        }

        if !self.has_resize && self.read_zero_expr.is_none() {
            walk_expr(self, e);
        }
    }
}
