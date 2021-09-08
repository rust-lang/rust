use crate::path_to_local_id;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, walk_expr, ErasedMap, NestedVisitorMap, Visitor};
use rustc_hir::{def::Res, Arm, Block, Body, BodyId, Destination, Expr, ExprKind, HirId, Stmt};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use std::ops::ControlFlow;

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

/// Calls the given function for each break expression.
pub fn visit_break_exprs<'tcx>(
    node: impl Visitable<'tcx>,
    f: impl FnMut(&'tcx Expr<'tcx>, Destination, Option<&'tcx Expr<'tcx>>),
) {
    struct V<F>(F);
    impl<'tcx, F: FnMut(&'tcx Expr<'tcx>, Destination, Option<&'tcx Expr<'tcx>>)> Visitor<'tcx> for V<F> {
        type Map = ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if let ExprKind::Break(dest, sub_expr) = e.kind {
                self.0(e, dest, sub_expr);
            }
            walk_expr(self, e);
        }
    }

    node.visit(&mut V(f));
}

/// Checks if the given resolved path is used in the given body.
pub fn is_res_used(cx: &LateContext<'_>, res: Res, body: BodyId) -> bool {
    struct V<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        res: Res,
        found: bool,
    }
    impl Visitor<'tcx> for V<'_, 'tcx> {
        type Map = Map<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.found {
                return;
            }

            if let ExprKind::Path(p) = &e.kind {
                if self.cx.qpath_res(p, e.hir_id) == self.res {
                    self.found = true;
                }
            } else {
                walk_expr(self, e);
            }
        }
    }

    let mut v = V { cx, res, found: false };
    v.visit_expr(&cx.tcx.hir().body(body).value);
    v.found
}

/// Calls the given function for each usage of the given local.
pub fn for_each_local_usage<'tcx, B>(
    cx: &LateContext<'tcx>,
    visitable: impl Visitable<'tcx>,
    id: HirId,
    f: impl FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>,
) -> ControlFlow<B> {
    struct V<'tcx, B, F> {
        map: Map<'tcx>,
        id: HirId,
        f: F,
        res: ControlFlow<B>,
    }
    impl<'tcx, B, F: FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>> Visitor<'tcx> for V<'tcx, B, F> {
        type Map = Map<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::OnlyBodies(self.map)
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.res.is_continue() {
                if path_to_local_id(e, self.id) {
                    self.res = (self.f)(e);
                } else {
                    walk_expr(self, e);
                }
            }
        }
    }

    let mut v = V {
        map: cx.tcx.hir(),
        id,
        f,
        res: ControlFlow::CONTINUE,
    };
    visitable.visit(&mut v);
    v.res
}

/// Checks if the given local is used.
pub fn is_local_used(cx: &LateContext<'tcx>, visitable: impl Visitable<'tcx>, id: HirId) -> bool {
    for_each_local_usage(cx, visitable, id, |_| ControlFlow::BREAK).is_break()
}
