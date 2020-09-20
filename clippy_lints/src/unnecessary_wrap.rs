use crate::utils::{
    is_type_diagnostic_item, match_qpath, multispan_sugg_with_applicability, paths, return_ty, snippet,
    span_lint_and_then,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{FnKind, NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::{hir::map::Map, ty::subst::GenericArgKind};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for private functions that only return `Ok` or `Some`.
    ///
    /// **Why is this bad?** It is not meaningful to wrap values when no `None` or `Err` is returned.
    ///
    /// **Known problems:** Since this lint changes function type signature, you may need to
    /// adjust some codes at callee side.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn get_cool_number(a: bool, b: bool) -> Option<i32> {
    ///     if a && b {
    ///         return Some(50);
    ///     }
    ///     if a {
    ///         Some(0)
    ///     } else {
    ///         Some(10)
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn get_cool_number(a: bool, b: bool) -> i32 {
    ///     if a && b {
    ///         return 50;
    ///     }
    ///     if a {
    ///         0
    ///     } else {
    ///         10
    ///     }
    /// }
    /// ```
    pub UNNECESSARY_WRAP,
    complexity,
    "functions that only return `Ok` or `Some`"
}

declare_lint_pass!(UnnecessaryWrap => [UNNECESSARY_WRAP]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryWrap {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &FnDecl<'tcx>,
        body: &Body<'tcx>,
        span: Span,
        hir_id: HirId,
    ) {
        if_chain! {
            if let FnKind::ItemFn(.., visibility, _) = fn_kind;
            if visibility.node.is_pub();
            then {
                return;
            }
        }

        if let ExprKind::Block(ref block, ..) = body.value.kind {
            let path = if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym!(option_type)) {
                &paths::OPTION_SOME
            } else if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym!(result_type)) {
                &paths::RESULT_OK
            } else {
                return;
            };

            let mut visitor = UnnecessaryWrapVisitor { result: Vec::new() };
            visitor.visit_block(block);
            let result = visitor.result;

            if result.iter().any(|expr| {
                if_chain! {
                    if let ExprKind::Call(ref func, ref args) = expr.kind;
                    if let ExprKind::Path(ref qpath) = func.kind;
                    if match_qpath(qpath, path);
                    if args.len() == 1;
                    then {
                        false
                    } else {
                        true
                    }
                }
            }) {
                return;
            }

            let suggs = result
                .iter()
                .filter_map(|expr| {
                    let snippet = if let ExprKind::Call(_, ref args) = expr.kind {
                        Some(snippet(cx, args[0].span, "..").to_string())
                    } else {
                        None
                    };
                    snippet.map(|snip| (expr.span, snip))
                })
                .chain({
                    let inner_ty = return_ty(cx, hir_id)
                        .walk()
                        .skip(1) // skip `std::option::Option` or `std::result::Result`
                        .take(1) // first outermost inner type is needed
                        .filter_map(|inner| match inner.unpack() {
                            GenericArgKind::Type(inner_ty) => Some(inner_ty.to_string()),
                            _ => None,
                        });
                    inner_ty.map(|inner_ty| (fn_decl.output.span(), inner_ty))
                });

            span_lint_and_then(
                cx,
                UNNECESSARY_WRAP,
                span,
                "this function returns unnecessarily wrapping data",
                move |diag| {
                    multispan_sugg_with_applicability(
                        diag,
                        "factor this out to",
                        Applicability::MachineApplicable,
                        suggs,
                    );
                },
            );
        }
    }
}

struct UnnecessaryWrapVisitor<'tcx> {
    result: Vec<&'tcx Expr<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for UnnecessaryWrapVisitor<'tcx> {
    type Map = Map<'tcx>;

    fn visit_block(&mut self, block: &'tcx Block<'tcx>) {
        for stmt in block.stmts {
            self.visit_stmt(stmt);
        }
        if let Some(expr) = block.expr {
            self.visit_expr(expr)
        }
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'tcx>) {
        match stmt.kind {
            StmtKind::Semi(ref expr) => {
                if let ExprKind::Ret(Some(value)) = expr.kind {
                    self.result.push(value);
                }
            },
            StmtKind::Expr(ref expr) => self.visit_expr(expr),
            _ => (),
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        match expr.kind {
            ExprKind::Ret(Some(value)) => self.result.push(value),
            ExprKind::Call(..) | ExprKind::Path(..) => self.result.push(expr),
            ExprKind::Block(ref block, _) | ExprKind::Loop(ref block, ..) => {
                self.visit_block(block);
            },
            ExprKind::Match(_, arms, _) => {
                for arm in arms {
                    self.visit_expr(arm.body);
                }
            },
            _ => intravisit::walk_expr(self, expr),
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
