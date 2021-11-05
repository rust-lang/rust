use crate::path_to_local_id;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{def::Res, Arm, Block, Body, BodyId, Expr, ExprKind, HirId, Stmt};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;

/// Convenience method for creating a `Visitor` with just `visit_expr` overridden and nested
/// bodies (i.e. closures) are visited.
/// If the callback returns `true`, the expr just provided to the callback is walked.
#[must_use]
pub fn expr_visitor<'tcx>(cx: &LateContext<'tcx>, f: impl FnMut(&'tcx Expr<'tcx>) -> bool) -> impl Visitor<'tcx> {
    struct V<'tcx, F> {
        hir: Map<'tcx>,
        f: F,
    }
    impl<'tcx, F: FnMut(&'tcx Expr<'tcx>) -> bool> Visitor<'tcx> for V<'tcx, F> {
        type Map = Map<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::OnlyBodies(self.hir)
        }

        fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
            if (self.f)(expr) {
                walk_expr(self, expr);
            }
        }
    }
    V { hir: cx.tcx.hir(), f }
}

/// Convenience method for creating a `Visitor` with just `visit_expr` overridden and nested
/// bodies (i.e. closures) are not visited.
/// If the callback returns `true`, the expr just provided to the callback is walked.
#[must_use]
pub fn expr_visitor_no_bodies<'tcx>(f: impl FnMut(&'tcx Expr<'tcx>) -> bool) -> impl Visitor<'tcx> {
    struct V<F>(F);
    impl<'tcx, F: FnMut(&'tcx Expr<'tcx>) -> bool> Visitor<'tcx> for V<F> {
        type Map = intravisit::ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if (self.0)(e) {
                walk_expr(self, e);
            }
        }
    }
    V(f)
}

/// returns `true` if expr contains match expr desugared from try
fn contains_try(expr: &hir::Expr<'_>) -> bool {
    let mut found = false;
    expr_visitor_no_bodies(|e| {
        if !found {
            found = matches!(e.kind, hir::ExprKind::Match(_, _, hir::MatchSource::TryDesugar));
        }
        !found
    })
    .visit_expr(expr);
    found
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
            intravisit::walk_stmt(&mut *self.inside_stmt(true), stmt);
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
                    hir::ExprKind::If(cond, then, else_opt) => {
                        self.inside_stmt(true).visit_expr(cond);
                        self.visit_expr(then);
                        if let Some(el) = else_opt {
                            self.visit_expr(el);
                        }
                    },
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

/// A type which can be visited.
pub trait Visitable<'tcx> {
    /// Calls the corresponding `visit_*` function on the visitor.
    fn visit<V: Visitor<'tcx>>(self, visitor: &mut V);
}
macro_rules! visitable_ref {
    ($t:ident, $f:ident) => {
        impl Visitable<'tcx> for &'tcx $t<'tcx> {
            fn visit<V: Visitor<'tcx>>(self, visitor: &mut V) {
                visitor.$f(self);
            }
        }
    };
}
visitable_ref!(Arm, visit_arm);
visitable_ref!(Block, visit_block);
visitable_ref!(Body, visit_body);
visitable_ref!(Expr, visit_expr);
visitable_ref!(Stmt, visit_stmt);

// impl<'tcx, I: IntoIterator> Visitable<'tcx> for I
// where
//     I::Item: Visitable<'tcx>,
// {
//     fn visit<V: Visitor<'tcx>>(self, visitor: &mut V) {
//         for x in self {
//             x.visit(visitor);
//         }
//     }
// }

/// Checks if the given resolved path is used in the given body.
pub fn is_res_used(cx: &LateContext<'_>, res: Res, body: BodyId) -> bool {
    let mut found = false;
    expr_visitor(cx, |e| {
        if found {
            return false;
        }

        if let ExprKind::Path(p) = &e.kind {
            if cx.qpath_res(p, e.hir_id) == res {
                found = true;
            }
        }
        !found
    })
    .visit_expr(&cx.tcx.hir().body(body).value);
    found
}

/// Checks if the given local is used.
pub fn is_local_used(cx: &LateContext<'tcx>, visitable: impl Visitable<'tcx>, id: HirId) -> bool {
    let mut is_used = false;
    let mut visitor = expr_visitor(cx, |expr| {
        if !is_used {
            is_used = path_to_local_id(expr, id);
        }
        !is_used
    });
    visitable.visit(&mut visitor);
    drop(visitor);
    is_used
}
