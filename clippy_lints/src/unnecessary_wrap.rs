use crate::utils::{is_type_diagnostic_item, match_qpath, paths, return_ty, snippet, span_lint_and_then};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{FnKind, Visitor};
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
    /// adjust some code at callee side.
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

        let (return_type, path) = if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym!(option_type)) {
            ("Option", &paths::OPTION_SOME)
        } else if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym!(result_type)) {
            ("Result", &paths::RESULT_OK)
        } else {
            return;
        };

        let mut suggs = Vec::new();
        let can_sugg = find_all_ret_expressions(cx, &body.value, |ret_expr| {
            if_chain! {
                if let ExprKind::Call(ref func, ref args) = ret_expr.kind;
                if let ExprKind::Path(ref qpath) = func.kind;
                if match_qpath(qpath, path);
                if args.len() == 1;
                then {
                    suggs.push((ret_expr.span, snippet(cx, args[0].span.source_callsite(), "..").to_string()));
                    true
                } else {
                    false
                }
            }
        });

        if can_sugg {
            span_lint_and_then(
                cx,
                UNNECESSARY_WRAP,
                span,
                "this function returns unnecessarily wrapping data",
                |diag| {
                    let inner_ty = return_ty(cx, hir_id)
                        .walk()
                        .skip(1) // skip `std::option::Option` or `std::result::Result`
                        .take(1) // take the first outermost inner type
                        .filter_map(|inner| match inner.unpack() {
                            GenericArgKind::Type(inner_ty) => Some(inner_ty.to_string()),
                            _ => None,
                        });
                    inner_ty.for_each(|inner_ty| {
                        diag.span_suggestion(
                            fn_decl.output.span(),
                            format!("remove `{}` from the return type...", return_type).as_str(),
                            inner_ty,
                            Applicability::MachineApplicable,
                        );
                    });
                    diag.multipart_suggestion(
                        "...and change the returning expressions",
                        suggs,
                        Applicability::MachineApplicable,
                    );
                },
            );
        }
    }
}

// code below is copied from `bind_instead_of_map`

fn find_all_ret_expressions<'hir, F>(_cx: &LateContext<'_>, expr: &'hir Expr<'hir>, callback: F) -> bool
where
    F: FnMut(&'hir Expr<'hir>) -> bool,
{
    struct RetFinder<F> {
        in_stmt: bool,
        failed: bool,
        cb: F,
    }

    struct WithStmtGuarg<'a, F> {
        val: &'a mut RetFinder<F>,
        prev_in_stmt: bool,
    }

    impl<F> RetFinder<F> {
        fn inside_stmt(&mut self, in_stmt: bool) -> WithStmtGuarg<'_, F> {
            let prev_in_stmt = std::mem::replace(&mut self.in_stmt, in_stmt);
            WithStmtGuarg {
                val: self,
                prev_in_stmt,
            }
        }
    }

    impl<F> std::ops::Deref for WithStmtGuarg<'_, F> {
        type Target = RetFinder<F>;

        fn deref(&self) -> &Self::Target {
            self.val
        }
    }

    impl<F> std::ops::DerefMut for WithStmtGuarg<'_, F> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.val
        }
    }

    impl<F> Drop for WithStmtGuarg<'_, F> {
        fn drop(&mut self) {
            self.val.in_stmt = self.prev_in_stmt;
        }
    }

    impl<'hir, F: FnMut(&'hir Expr<'hir>) -> bool> intravisit::Visitor<'hir> for RetFinder<F> {
        type Map = Map<'hir>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_stmt(&mut self, stmt: &'hir Stmt<'_>) {
            intravisit::walk_stmt(&mut *self.inside_stmt(true), stmt)
        }

        fn visit_expr(&mut self, expr: &'hir Expr<'_>) {
            if self.failed {
                return;
            }
            if self.in_stmt {
                match expr.kind {
                    ExprKind::Ret(Some(expr)) => self.inside_stmt(false).visit_expr(expr),
                    _ => intravisit::walk_expr(self, expr),
                }
            } else {
                match expr.kind {
                    ExprKind::Match(cond, arms, _) => {
                        self.inside_stmt(true).visit_expr(cond);
                        for arm in arms {
                            self.visit_expr(arm.body);
                        }
                    },
                    ExprKind::Block(..) => intravisit::walk_expr(self, expr),
                    ExprKind::Ret(Some(expr)) => self.visit_expr(expr),
                    _ => self.failed |= !(self.cb)(expr),
                }
            }
        }
    }

    !contains_try(expr) && {
        let mut ret_finder = RetFinder {
            in_stmt: false,
            failed: false,
            cb: callback,
        };
        ret_finder.visit_expr(expr);
        !ret_finder.failed
    }
}

/// returns `true` if expr contains match expr desugared from try
fn contains_try(expr: &Expr<'_>) -> bool {
    struct TryFinder {
        found: bool,
    }

    impl<'hir> intravisit::Visitor<'hir> for TryFinder {
        type Map = Map<'hir>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_expr(&mut self, expr: &'hir Expr<'hir>) {
            if self.found {
                return;
            }
            match expr.kind {
                ExprKind::Match(_, _, MatchSource::TryDesugar) => self.found = true,
                _ => intravisit::walk_expr(self, expr),
            }
        }
    }

    let mut visitor = TryFinder { found: false };
    visitor.visit_expr(expr);
    visitor.found
}
