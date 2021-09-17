use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{
    match_def_path, path_to_local_id, paths, peel_hir_expr_while, ty::is_uninit_value_valid_for_ty, SpanlessEq,
};
use rustc_hir::def::Res;
use rustc_hir::{Block, Expr, ExprKind, HirId, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the creation of uninitialized `Vec<T>` by calling `set_len()`
    /// immediately after `with_capacity()` or `reserve()`.
    ///
    /// ### Why is this bad?
    /// It creates `Vec<T>` that contains uninitialized data, which leads to an
    /// undefined behavior with most safe operations.
    /// Notably, using uninitialized `Vec<u8>` with generic `Read` is unsound.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let mut vec: Vec<u8> = Vec::with_capacity(1000);
    /// unsafe { vec.set_len(1000); }
    /// reader.read(&mut vec); // undefined behavior!
    /// ```
    /// Use an initialized buffer:
    /// ```rust,ignore
    /// let mut vec: Vec<u8> = vec![0; 1000];
    /// reader.read(&mut vec);
    /// ```
    /// Or, wrap the content in `MaybeUninit`:
    /// ```rust,ignore
    /// let mut vec: Vec<MaybeUninit<T>> = Vec::with_capacity(1000);
    /// unsafe { vec.set_len(1000); }
    /// ```
    pub UNINIT_VEC,
    correctness,
    "Vec with uninitialized data"
}

declare_lint_pass!(UninitVec => [UNINIT_VEC]);

impl<'tcx> LateLintPass<'tcx> for UninitVec {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        for w in block.stmts.windows(2) {
            if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = w[1].kind {
                handle_uninit_vec_pair(cx, &w[0], expr);
            }
        }

        if let (Some(stmt), Some(expr)) = (block.stmts.last(), block.expr) {
            handle_uninit_vec_pair(cx, stmt, expr);
        }
    }
}

fn handle_uninit_vec_pair(
    cx: &LateContext<'tcx>,
    maybe_with_capacity_or_reserve: &'tcx Stmt<'tcx>,
    maybe_set_len: &'tcx Expr<'tcx>,
) {
    if_chain! {
        if let Some(vec) = extract_with_capacity_or_reserve_target(cx, maybe_with_capacity_or_reserve);
        if let Some((set_len_self, call_span)) = extract_set_len_self(cx, maybe_set_len);
        if vec.eq_expr(cx, set_len_self);
        if let ty::Ref(_, vec_ty, _) = cx.typeck_results().expr_ty_adjusted(set_len_self).kind();
        if let ty::Adt(_, substs) = vec_ty.kind();
        // Check T of Vec<T>
        if !is_uninit_value_valid_for_ty(cx, substs.type_at(0));
        then {
            span_lint_and_note(
                cx,
                UNINIT_VEC,
                call_span,
                "calling `set_len()` immediately after reserving a buffer creates uninitialized values",
                Some(maybe_with_capacity_or_reserve.span),
                "the buffer is reserved here"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum LocalOrExpr<'tcx> {
    Local(HirId),
    Expr(&'tcx Expr<'tcx>),
}

impl<'tcx> LocalOrExpr<'tcx> {
    fn eq_expr(self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
        match self {
            LocalOrExpr::Local(hir_id) => path_to_local_id(expr, hir_id),
            LocalOrExpr::Expr(self_expr) => SpanlessEq::new(cx).eq_expr(self_expr, expr),
        }
    }
}

/// Returns the target vec of `Vec::with_capacity()` or `Vec::reserve()`
fn extract_with_capacity_or_reserve_target(cx: &LateContext<'_>, stmt: &'tcx Stmt<'_>) -> Option<LocalOrExpr<'tcx>> {
    match stmt.kind {
        StmtKind::Local(local) => {
            // let mut x = Vec::with_capacity()
            if_chain! {
                if let Some(init_expr) = local.init;
                if let PatKind::Binding(_, hir_id, _, None) = local.pat.kind;
                if is_with_capacity(cx, init_expr);
                then {
                    Some(LocalOrExpr::Local(hir_id))
                } else {
                    None
                }
            }
        },
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
            match expr.kind {
                ExprKind::Assign(lhs, rhs, _span) if is_with_capacity(cx, rhs) => {
                    // self.vec = Vec::with_capacity()
                    Some(LocalOrExpr::Expr(lhs))
                },
                ExprKind::MethodCall(path, _, [self_expr, _], _) => {
                    // self.vec.reserve()
                    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(self_expr).peel_refs(), sym::vec_type)
                        && path.ident.name.as_str() == "reserve"
                    {
                        Some(LocalOrExpr::Expr(self_expr))
                    } else {
                        None
                    }
                },
                _ => None,
            }
        },
        StmtKind::Item(_) => None,
    }
}

fn is_with_capacity(cx: &LateContext<'_>, expr: &'tcx Expr<'_>) -> bool {
    if_chain! {
        if let ExprKind::Call(path_expr, _) = &expr.kind;
        if let ExprKind::Path(qpath) = &path_expr.kind;
        if let Res::Def(_, def_id) = cx.qpath_res(qpath, path_expr.hir_id);
        then {
            match_def_path(cx, def_id, &paths::VEC_WITH_CAPACITY)
        } else {
            false
        }
    }
}

/// Returns self if the expression is `Vec::set_len()`
fn extract_set_len_self(cx: &LateContext<'_>, expr: &'tcx Expr<'_>) -> Option<(&'tcx Expr<'tcx>, Span)> {
    // peel unsafe blocks in `unsafe { vec.set_len() }`
    let expr = peel_hir_expr_while(expr, |e| {
        if let ExprKind::Block(block, _) = e.kind {
            match (block.stmts.get(0).map(|stmt| &stmt.kind), block.expr) {
                (None, Some(expr)) => Some(expr),
                (Some(StmtKind::Expr(expr) | StmtKind::Semi(expr)), None) => Some(expr),
                _ => None,
            }
        } else {
            None
        }
    });
    match expr.kind {
        ExprKind::MethodCall(_, _, [vec_expr, _], _) => {
            cx.typeck_results().type_dependent_def_id(expr.hir_id).and_then(|id| {
                if match_def_path(cx, id, &paths::VEC_SET_LEN) {
                    Some((vec_expr, expr.span))
                } else {
                    None
                }
            })
        },
        _ => None,
    }
}
