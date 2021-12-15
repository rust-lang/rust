use crate::path_to_local_id;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{self, walk_block, walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{
    Arm, Block, BlockCheckMode, Body, BodyId, Expr, ExprKind, HirId, ItemId, ItemKind, Stmt, UnOp, Unsafety,
};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_middle::ty;

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

/// Checks if the given expression is a constant.
pub fn is_const_evaluatable(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> bool {
    struct V<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        is_const: bool,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        type Map = Map<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if !self.is_const {
                return;
            }
            match e.kind {
                ExprKind::ConstBlock(_) => return,
                ExprKind::Call(
                    &Expr {
                        kind: ExprKind::Path(ref p),
                        hir_id,
                        ..
                    },
                    _,
                ) if self
                    .cx
                    .qpath_res(p, hir_id)
                    .opt_def_id()
                    .map_or(false, |id| self.cx.tcx.is_const_fn_raw(id)) => {},
                ExprKind::MethodCall(..)
                    if self
                        .cx
                        .typeck_results()
                        .type_dependent_def_id(e.hir_id)
                        .map_or(false, |id| self.cx.tcx.is_const_fn_raw(id)) => {},
                ExprKind::Binary(_, lhs, rhs)
                    if self.cx.typeck_results().expr_ty(lhs).peel_refs().is_primitive_ty()
                        && self.cx.typeck_results().expr_ty(rhs).peel_refs().is_primitive_ty() => {},
                ExprKind::Unary(UnOp::Deref, e) if self.cx.typeck_results().expr_ty(e).is_ref() => (),
                ExprKind::Unary(_, e) if self.cx.typeck_results().expr_ty(e).peel_refs().is_primitive_ty() => (),
                ExprKind::Index(base, _)
                    if matches!(
                        self.cx.typeck_results().expr_ty(base).peel_refs().kind(),
                        ty::Slice(_) | ty::Array(..)
                    ) => {},
                ExprKind::Path(ref p)
                    if matches!(
                        self.cx.qpath_res(p, e.hir_id),
                        Res::Def(
                            DefKind::Const
                                | DefKind::AssocConst
                                | DefKind::AnonConst
                                | DefKind::ConstParam
                                | DefKind::Ctor(..)
                                | DefKind::Fn
                                | DefKind::AssocFn,
                            _
                        ) | Res::SelfCtor(_)
                    ) => {},

                ExprKind::AddrOf(..)
                | ExprKind::Array(_)
                | ExprKind::Block(..)
                | ExprKind::Cast(..)
                | ExprKind::DropTemps(_)
                | ExprKind::Field(..)
                | ExprKind::If(..)
                | ExprKind::Let(..)
                | ExprKind::Lit(_)
                | ExprKind::Match(..)
                | ExprKind::Repeat(..)
                | ExprKind::Struct(..)
                | ExprKind::Tup(_)
                | ExprKind::Type(..) => (),

                _ => {
                    self.is_const = false;
                    return;
                },
            }
            walk_expr(self, e);
        }
    }

    let mut v = V { cx, is_const: true };
    v.visit_expr(e);
    v.is_const
}

/// Checks if the given expression performs an unsafe operation outside of an unsafe block.
pub fn is_expr_unsafe(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> bool {
    struct V<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        is_unsafe: bool,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        type Map = Map<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
        }
        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.is_unsafe {
                return;
            }
            match e.kind {
                ExprKind::Unary(UnOp::Deref, e) if self.cx.typeck_results().expr_ty(e).is_unsafe_ptr() => {
                    self.is_unsafe = true;
                },
                ExprKind::MethodCall(..)
                    if self
                        .cx
                        .typeck_results()
                        .type_dependent_def_id(e.hir_id)
                        .map_or(false, |id| self.cx.tcx.fn_sig(id).unsafety() == Unsafety::Unsafe) =>
                {
                    self.is_unsafe = true;
                },
                ExprKind::Call(func, _) => match *self.cx.typeck_results().expr_ty(func).peel_refs().kind() {
                    ty::FnDef(id, _) if self.cx.tcx.fn_sig(id).unsafety() == Unsafety::Unsafe => self.is_unsafe = true,
                    ty::FnPtr(sig) if sig.unsafety() == Unsafety::Unsafe => self.is_unsafe = true,
                    _ => walk_expr(self, e),
                },
                ExprKind::Path(ref p)
                    if self
                        .cx
                        .qpath_res(p, e.hir_id)
                        .opt_def_id()
                        .map_or(false, |id| self.cx.tcx.is_mutable_static(id)) =>
                {
                    self.is_unsafe = true;
                },
                _ => walk_expr(self, e),
            }
        }
        fn visit_block(&mut self, b: &'tcx Block<'_>) {
            if !matches!(b.rules, BlockCheckMode::UnsafeBlock(_)) {
                walk_block(self, b);
            }
        }
        fn visit_nested_item(&mut self, id: ItemId) {
            if let ItemKind::Impl(i) = &self.cx.tcx.hir().item(id).kind {
                self.is_unsafe = i.unsafety == Unsafety::Unsafe;
            }
        }
    }
    let mut v = V { cx, is_unsafe: false };
    v.visit_expr(e);
    v.is_unsafe
}
