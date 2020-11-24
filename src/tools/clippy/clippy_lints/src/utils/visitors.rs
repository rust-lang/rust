use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;

/// returns `true` if expr contains match expr desugared from try
fn contains_try(expr: &hir::Expr<'_>) -> bool {
    struct TryFinder {
        found: bool,
    }

    impl<'hir> intravisit::Visitor<'hir> for TryFinder {
        type Map = Map<'hir>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
            if self.found {
                return;
            }
            match expr.kind {
                hir::ExprKind::Match(_, _, hir::MatchSource::TryDesugar) => self.found = true,
                _ => intravisit::walk_expr(self, expr),
            }
        }
    }

    let mut visitor = TryFinder { found: false };
    visitor.visit_expr(expr);
    visitor.found
}

pub fn find_all_ret_expressions<'hir, F>(_cx: &LateContext<'_>, expr: &'hir hir::Expr<'hir>, callback: F) -> bool
where
    F: FnMut(&'hir hir::Expr<'hir>) -> bool,
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

    impl<'hir, F: FnMut(&'hir hir::Expr<'hir>) -> bool> intravisit::Visitor<'hir> for RetFinder<F> {
        type Map = Map<'hir>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_stmt(&mut self, stmt: &'hir hir::Stmt<'_>) {
            intravisit::walk_stmt(&mut *self.inside_stmt(true), stmt)
        }

        fn visit_expr(&mut self, expr: &'hir hir::Expr<'_>) {
            if self.failed {
                return;
            }
            if self.in_stmt {
                match expr.kind {
                    hir::ExprKind::Ret(Some(expr)) => self.inside_stmt(false).visit_expr(expr),
                    _ => intravisit::walk_expr(self, expr),
                }
            } else {
                match expr.kind {
                    hir::ExprKind::Match(cond, arms, _) => {
                        self.inside_stmt(true).visit_expr(cond);
                        for arm in arms {
                            self.visit_expr(arm.body);
                        }
                    },
                    hir::ExprKind::Block(..) => intravisit::walk_expr(self, expr),
                    hir::ExprKind::Ret(Some(expr)) => self.visit_expr(expr),
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
