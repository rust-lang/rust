use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::{get_vec_init_kind, VecInitKind};
use clippy_utils::source::snippet;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, path_to_local_id};
use core::ops::ControlFlow;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{
    BindingAnnotation, Block, Expr, ExprKind, HirId, Local, Mutability, PatKind, QPath, Stmt, StmtKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Informs the user about a more concise way to create a vector with a known capacity.
    ///
    /// ### Why is this bad?
    /// The `Vec::with_capacity` constructor is easier to understand.
    ///
    /// ### Example
    /// ```rust
    /// {
    ///     let mut v = vec![];
    ///     v.reserve(space_hint);
    ///     v
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// Vec::with_capacity(space_hint)
    /// ```
    #[clippy::version = "1.73.0"]
    pub RESERVE_AFTER_INITIALIZATION,
    complexity,
    "`reserve` called immediatly after `Vec` creation"
}
impl_lint_pass!(ReserveAfterInitialization => [RESERVE_AFTER_INITIALIZATION]);

/*impl<'tcx> LateLintPass<'tcx> for ReserveAfterInitialization {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx rustc_hir::Expr<'_>) {
        if let ExprKind::Assign(left, right, _) = expr.kind
        && let ExprKind::Path(QPath::Resolved(None, path)) = left.kind
        && let [name] = &path.segments
        && let Res::Local(id) = path.res
        && let Some(init) = get_vec_init_kind(cx, right)
        && !matches!(init, VecInitKind::WithExprCapacity(_)) {
            span_lint_and_help(
                cx,
                RESERVE_AFTER_INITIALIZATION,
                expr.span,
                "`reserve` called just after the initialisation of the vector",
                None,
                "use `Vec::with_capacity(space_hint)` instead"
            );
        }
    }

    /*fn check_block(&mut self, cx: &LateContext<'_>, block: &'_ rustc_hir::Block<'_>) {
        span_lint_and_help(
            cx,
            RESERVE_AFTER_INITIALIZATION,
            block.span,
            "`reserve` called just after the initialisation of the vector",
            None,
            "use `Vec::with_capacity(space_hint)` instead"
        );
    }*/
}*/

#[derive(Default)]
pub struct ReserveAfterInitialization {
    searcher: Option<VecReserveSearcher>,
}

struct VecReserveSearcher {
    local_id: HirId,
    lhs_is_let: bool,
    let_ty_span: Option<Span>,
    name: Symbol,
    err_span: Span,
    last_reserve_expr: HirId,
    space_hint: usize,
}
impl VecReserveSearcher {
    fn display_err(&self, cx: &LateContext<'_>) {
        if self.space_hint == 0 {
            return;
        }

        let mut needs_mut = false;
        let _res = for_each_local_use_after_expr(cx, self.local_id, self.last_reserve_expr, |e| {
            let Some(parent) = get_parent_expr(cx, e) else {
                return ControlFlow::Continue(());
            };
            let adjusted_ty = cx.typeck_results().expr_ty_adjusted(e);
            let adjusted_mut = adjusted_ty.ref_mutability().unwrap_or(Mutability::Not);
            needs_mut |= adjusted_mut == Mutability::Mut;
            match parent.kind {
                ExprKind::AddrOf(_, Mutability::Mut, _) => {
                    needs_mut = true;
                    return ControlFlow::Break(true);
                },
                ExprKind::Unary(UnOp::Deref, _) | ExprKind::Index(..) if !needs_mut => {
                    let mut last_place = parent;
                    while let Some(parent) = get_parent_expr(cx, last_place) {
                        if matches!(parent.kind, ExprKind::Unary(UnOp::Deref, _) | ExprKind::Field(..))
                            || matches!(parent.kind, ExprKind::Index(e, _, _) if e.hir_id == last_place.hir_id)
                        {
                            last_place = parent;
                        } else {
                            break;
                        }
                    }
                    needs_mut |= cx.typeck_results().expr_ty_adjusted(last_place).ref_mutability()
                        == Some(Mutability::Mut)
                        || get_parent_expr(cx, last_place)
                            .map_or(false, |e| matches!(e.kind, ExprKind::AddrOf(_, Mutability::Mut, _)));
                },
                ExprKind::MethodCall(_, recv, ..)
                    if recv.hir_id == e.hir_id
                        && adjusted_mut == Mutability::Mut
                        && !adjusted_ty.peel_refs().is_slice() =>
                {
                    // No need to set `needs_mut` to true. The receiver will be either explicitly borrowed, or it will
                    // be implicitly borrowed via an adjustment. Both of these cases are already handled by this point.
                    return ControlFlow::Break(true);
                },
                ExprKind::Assign(lhs, ..) if e.hir_id == lhs.hir_id => {
                    needs_mut = true;
                    return ControlFlow::Break(false);
                },
                _ => (),
            }
            ControlFlow::Continue(())
        });

        let mut s = if self.lhs_is_let {
            String::from("let ")
        } else {
            String::new()
        };
        if needs_mut {
            s.push_str("mut ");
        }
        s.push_str(self.name.as_str());
        if let Some(span) = self.let_ty_span {
            s.push_str(": ");
            s.push_str(&snippet(cx, span, "_"));
        }
        s.push_str(format!(" = Vec::with_capacity({});", self.space_hint).as_str());

        span_lint_and_sugg(
            cx,
            RESERVE_AFTER_INITIALIZATION,
            self.err_span,
            "calls to `reverse` immediately after creation",
            "consider using `Vec::with_capacity(space_hint)`",
            s,
            Applicability::HasPlaceholders,
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for ReserveAfterInitialization {
    fn check_block(&mut self, _: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        self.searcher = None;
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        if let Some(init_expr) = local.init
            && let PatKind::Binding(BindingAnnotation::MUT, id, name, None) = local.pat.kind
            && !in_external_macro(cx.sess(), local.span)
            && let Some(init) = get_vec_init_kind(cx, init_expr)
            && !matches!(init, VecInitKind::WithExprCapacity(_))
        {
            self.searcher = Some(VecReserveSearcher {
                local_id: id,
                lhs_is_let: true,
                name: name.name,
                let_ty_span: local.ty.map(|ty| ty.span),
                err_span: local.span,
                last_reserve_expr: init_expr.hir_id,
                space_hint: 0
            });
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if self.searcher.is_none()
            && let ExprKind::Assign(left, right, _) = expr.kind
            && let ExprKind::Path(QPath::Resolved(None, path)) = left.kind
            && let [name] = &path.segments
            && let Res::Local(id) = path.res
            && !in_external_macro(cx.sess(), expr.span)
            && let Some(init) = get_vec_init_kind(cx, right)
            && !matches!(init, VecInitKind::WithExprCapacity(_))
        {
            self.searcher = Some(VecReserveSearcher {
                local_id: id,
                lhs_is_let: false,
                let_ty_span: None,
                name: name.ident.name,
                err_span: expr.span,
                last_reserve_expr: expr.hir_id,
                space_hint: 0
            });
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let Some(searcher) = self.searcher.take() {
            if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind
                && let ExprKind::MethodCall(name, self_arg, other_args, _) = expr.kind
                && other_args.len() == 1
                && let ExprKind::Lit(lit) = other_args[0].kind
                && let LitKind::Int(space_hint, _) = lit.node
                && path_to_local_id(self_arg, searcher.local_id)
                && name.ident.as_str() == "reserve"
            {
                self.searcher = Some(VecReserveSearcher {
                    err_span: searcher.err_span.to(stmt.span),
                    last_reserve_expr: expr.hir_id,
                    space_hint: space_hint as usize,
                    .. searcher
                });
            } else {
                searcher.display_err(cx);
            }
        }
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        if let Some(searcher) = self.searcher.take() {
            searcher.display_err(cx);
        }
    }
}
