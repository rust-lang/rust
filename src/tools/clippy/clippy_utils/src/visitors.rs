use crate::ty::needs_ordered_drop;
use crate::{get_enclosing_block, path_to_local_id};
use core::ops::ControlFlow::{self, Break, Continue};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::intravisit::{self, walk_block, walk_expr, Visitor};
use rustc_hir::{
    AnonConst, Arm, Block, BlockCheckMode, Body, BodyId, Expr, ExprKind, HirId, ItemId, ItemKind, Let, Pat, QPath,
    Stmt, UnOp, UnsafeSource, Unsafety,
};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeckResults};
use rustc_span::Span;

mod internal {
    /// Trait for visitor functions to control whether or not to descend to child nodes. Implemented
    /// for only two types. `()` always descends. `Descend` allows controlled descent.
    pub trait MaybeDescend {
        fn descend(&self) -> bool;
    }
}
use internal::MaybeDescend;

impl MaybeDescend for () {
    fn descend(&self) -> bool {
        true
    }
}

/// Allows for controlled descent when using visitor functions. Use `()` instead when always
/// descending into child nodes.
#[derive(Clone, Copy)]
pub enum Descend {
    Yes,
    No,
}
impl From<bool> for Descend {
    fn from(from: bool) -> Self {
        if from { Self::Yes } else { Self::No }
    }
}
impl MaybeDescend for Descend {
    fn descend(&self) -> bool {
        matches!(self, Self::Yes)
    }
}

/// A type which can be visited.
pub trait Visitable<'tcx> {
    /// Calls the corresponding `visit_*` function on the visitor.
    fn visit<V: Visitor<'tcx>>(self, visitor: &mut V) -> ControlFlow<V::BreakTy>;
}
macro_rules! visitable_ref {
    ($t:ident, $f:ident) => {
        impl<'tcx> Visitable<'tcx> for &'tcx $t<'tcx> {
            fn visit<V: Visitor<'tcx>>(self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
                visitor.$f(self)
            }
        }
    };
}
visitable_ref!(Arm, visit_arm);
visitable_ref!(Block, visit_block);
visitable_ref!(Body, visit_body);
visitable_ref!(Expr, visit_expr);
visitable_ref!(Stmt, visit_stmt);

/// Calls the given function once for each expression contained. This does not enter any bodies or
/// nested items.
pub fn for_each_expr<'tcx, B, C: MaybeDescend>(
    node: impl Visitable<'tcx>,
    f: impl FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B, C>,
) -> Option<B> {
    struct V<F> {
        f: F,
    }
    impl<'tcx, B, C: MaybeDescend, F: FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B, C>> Visitor<'tcx> for V<F> {
        type BreakTy = B;
        fn visit_expr(&mut self, e: &'tcx Expr<'tcx>) -> ControlFlow<B> {
            if (self.f)(e)?.descend() {
                walk_expr(self, e)
            } else {
                Continue(())
            }
        }

        // Avoid unnecessary `walk_*` calls.
        fn visit_ty(&mut self, _: &'tcx hir::Ty<'tcx>) -> ControlFlow<B> {
            Continue(())
        }
        fn visit_pat(&mut self, _: &'tcx Pat<'tcx>) -> ControlFlow<B> {
            Continue(())
        }
        fn visit_qpath(&mut self, _: &'tcx QPath<'tcx>, _: HirId, _: Span) -> ControlFlow<B> {
            Continue(())
        }
        // Avoid monomorphising all `visit_*` functions.
        fn visit_nested_item(&mut self, _: ItemId) -> ControlFlow<B> {
            Continue(())
        }
    }
    let mut v = V { f };
    node.visit(&mut v).break_value()
}

/// Calls the given function once for each expression contained. This will enter bodies, but not
/// nested items.
pub fn for_each_expr_with_closures<'tcx, B, C: MaybeDescend>(
    cx: &LateContext<'tcx>,
    node: impl Visitable<'tcx>,
    f: impl FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B, C>,
) -> Option<B> {
    struct V<'tcx, F> {
        tcx: TyCtxt<'tcx>,
        f: F,
    }
    impl<'tcx, B, C: MaybeDescend, F: FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B, C>> Visitor<'tcx> for V<'tcx, F> {
        type NestedFilter = nested_filter::OnlyBodies;
        type BreakTy = B;
        fn nested_visit_map(&mut self) -> Self::Map {
            self.tcx.hir()
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'tcx>) -> ControlFlow<B> {
            if (self.f)(e)?.descend() {
                walk_expr(self, e)
            } else {
                Continue(())
            }
        }

        // Only walk closures
        fn visit_anon_const(&mut self, _: &'tcx AnonConst) -> ControlFlow<B> {
            Continue(())
        }
        // Avoid unnecessary `walk_*` calls.
        fn visit_ty(&mut self, _: &'tcx hir::Ty<'tcx>) -> ControlFlow<B> {
            Continue(())
        }
        fn visit_pat(&mut self, _: &'tcx Pat<'tcx>) -> ControlFlow<B> {
            Continue(())
        }
        fn visit_qpath(&mut self, _: &'tcx QPath<'tcx>, _: HirId, _: Span) -> ControlFlow<B> {
            Continue(())
        }
        // Avoid monomorphising all `visit_*` functions.
        fn visit_nested_item(&mut self, _: ItemId) -> ControlFlow<B> {
            Continue(())
        }
    }
    let mut v = V { tcx: cx.tcx, f };
    node.visit(&mut v).break_value()
}

/// returns `true` if expr contains match expr desugared from try
fn contains_try(expr: &hir::Expr<'_>) -> bool {
    for_each_expr(expr, |e| {
        if matches!(e.kind, hir::ExprKind::Match(_, _, hir::MatchSource::TryDesugar)) {
            Break(())
        } else {
            Continue(())
        }
    })
    .is_some()
}

pub fn find_all_ret_expressions<'hir, F>(_cx: &LateContext<'_>, expr: &'hir hir::Expr<'hir>, callback: F) -> bool
where
    F: FnMut(&'hir hir::Expr<'hir>) -> bool,
{
    struct RetFinder<F> {
        in_stmt: bool,
        cb: F,
    }

    struct WithStmtGuard<'a, F> {
        val: &'a mut RetFinder<F>,
        prev_in_stmt: bool,
    }

    impl<F> RetFinder<F> {
        fn inside_stmt(&mut self, in_stmt: bool) -> WithStmtGuard<'_, F> {
            let prev_in_stmt = std::mem::replace(&mut self.in_stmt, in_stmt);
            WithStmtGuard {
                val: self,
                prev_in_stmt,
            }
        }
    }

    impl<F> std::ops::Deref for WithStmtGuard<'_, F> {
        type Target = RetFinder<F>;

        fn deref(&self) -> &Self::Target {
            self.val
        }
    }

    impl<F> std::ops::DerefMut for WithStmtGuard<'_, F> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.val
        }
    }

    impl<F> Drop for WithStmtGuard<'_, F> {
        fn drop(&mut self) {
            self.val.in_stmt = self.prev_in_stmt;
        }
    }

    impl<'hir, F: FnMut(&'hir hir::Expr<'hir>) -> bool> intravisit::Visitor<'hir> for RetFinder<F> {
        type BreakTy = ();
        fn visit_stmt(&mut self, stmt: &'hir hir::Stmt<'_>) -> ControlFlow<()> {
            intravisit::walk_stmt(&mut *self.inside_stmt(true), stmt)
        }

        fn visit_expr(&mut self, expr: &'hir hir::Expr<'_>) -> ControlFlow<()> {
            if self.in_stmt {
                match expr.kind {
                    hir::ExprKind::Ret(Some(expr)) => self.inside_stmt(false).visit_expr(expr),
                    _ => intravisit::walk_expr(self, expr),
                }
            } else {
                match expr.kind {
                    hir::ExprKind::If(cond, then, else_opt) => {
                        self.inside_stmt(true).visit_expr(cond)?;
                        self.visit_expr(then)?;
                        if let Some(el) = else_opt {
                            self.visit_expr(el)?;
                        }
                        Continue(())
                    },
                    hir::ExprKind::Match(cond, arms, _) => {
                        self.inside_stmt(true).visit_expr(cond)?;
                        for arm in arms {
                            self.visit_expr(arm.body)?;
                        }
                        Continue(())
                    },
                    hir::ExprKind::Block(..) => intravisit::walk_expr(self, expr),
                    hir::ExprKind::Ret(Some(expr)) => self.visit_expr(expr),
                    _ if !(self.cb)(expr) => Break(()),
                    _ => Continue(()),
                }
            }
        }
    }

    !contains_try(expr) && {
        let mut ret_finder = RetFinder {
            in_stmt: false,
            cb: callback,
        };
        ret_finder.visit_expr(expr).is_continue()
    }
}

/// Checks if the given resolved path is used in the given body.
pub fn is_res_used(cx: &LateContext<'_>, res: Res, body: BodyId) -> bool {
    for_each_expr_with_closures(cx, cx.tcx.hir().body(body).value, |e| {
        if let ExprKind::Path(p) = &e.kind {
            if cx.qpath_res(p, e.hir_id) == res {
                return Break(());
            }
        }
        Continue(())
    })
    .is_some()
}

/// Checks if the given local is used.
pub fn is_local_used<'tcx>(cx: &LateContext<'tcx>, visitable: impl Visitable<'tcx>, id: HirId) -> bool {
    for_each_expr_with_closures(cx, visitable, |e| {
        if path_to_local_id(e, id) {
            Break(())
        } else {
            Continue(())
        }
    })
    .is_some()
}

/// Checks if the given expression is a constant.
pub fn is_const_evaluatable<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> bool {
    struct V<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        type NestedFilter = nested_filter::OnlyBodies;
        type BreakTy = ();
        fn nested_visit_map(&mut self) -> Self::Map {
            self.cx.tcx.hir()
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) -> ControlFlow<()> {
            match e.kind {
                ExprKind::ConstBlock(_) => return Continue(()),
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

                _ => return Break(()),
            }
            walk_expr(self, e)
        }
    }

    let mut v = V { cx };
    v.visit_expr(e).is_continue()
}

/// Checks if the given expression performs an unsafe operation outside of an unsafe block.
pub fn is_expr_unsafe<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> bool {
    struct V<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        type NestedFilter = nested_filter::OnlyBodies;
        type BreakTy = ();
        fn nested_visit_map(&mut self) -> Self::Map {
            self.cx.tcx.hir()
        }
        fn visit_expr(&mut self, e: &'tcx Expr<'_>) -> ControlFlow<()> {
            match e.kind {
                ExprKind::Unary(UnOp::Deref, e) if self.cx.typeck_results().expr_ty(e).is_unsafe_ptr() => Break(()),
                ExprKind::MethodCall(..)
                    if self
                        .cx
                        .typeck_results()
                        .type_dependent_def_id(e.hir_id)
                        .map_or(false, |id| {
                            self.cx.tcx.fn_sig(id).skip_binder().unsafety() == Unsafety::Unsafe
                        }) =>
                {
                    Break(())
                },
                ExprKind::Call(func, _) => match *self.cx.typeck_results().expr_ty(func).peel_refs().kind() {
                    ty::FnDef(id, _) if self.cx.tcx.fn_sig(id).skip_binder().unsafety() == Unsafety::Unsafe => {
                        Break(())
                    },
                    ty::FnPtr(sig) if sig.unsafety() == Unsafety::Unsafe => Break(()),
                    _ => walk_expr(self, e),
                },
                ExprKind::Path(ref p)
                    if self
                        .cx
                        .qpath_res(p, e.hir_id)
                        .opt_def_id()
                        .map_or(false, |id| self.cx.tcx.is_mutable_static(id)) =>
                {
                    Break(())
                },
                _ => walk_expr(self, e),
            }
        }
        fn visit_block(&mut self, b: &'tcx Block<'_>) -> ControlFlow<()> {
            if !matches!(b.rules, BlockCheckMode::UnsafeBlock(_)) {
                walk_block(self, b)
            } else {
                Continue(())
            }
        }
        fn visit_nested_item(&mut self, id: ItemId) -> ControlFlow<()> {
            if let ItemKind::Impl(i) = &self.cx.tcx.hir().item(id).kind
                && i.unsafety == Unsafety::Unsafe
            {
                Break(())
            } else {
                Continue(())
            }
        }
    }
    let mut v = V { cx };
    v.visit_expr(e).is_break()
}

/// Checks if the given expression contains an unsafe block
pub fn contains_unsafe_block<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) -> bool {
    struct V<'cx, 'tcx> {
        cx: &'cx LateContext<'tcx>,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        type NestedFilter = nested_filter::OnlyBodies;
        type BreakTy = ();
        fn nested_visit_map(&mut self) -> Self::Map {
            self.cx.tcx.hir()
        }

        fn visit_block(&mut self, b: &'tcx Block<'_>) -> ControlFlow<()> {
            if b.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) {
                return Break(());
            }
            walk_block(self, b)
        }
    }
    let mut v = V { cx };
    v.visit_expr(e).is_break()
}

/// Runs the given function for each sub-expression producing the final value consumed by the parent
/// of the give expression.
///
/// e.g. for the following expression
/// ```rust,ignore
/// if foo {
///     f(0)
/// } else {
///     1 + 1
/// }
/// ```
/// this will pass both `f(0)` and `1+1` to the given function.
pub fn for_each_value_source<'tcx, B>(
    e: &'tcx Expr<'tcx>,
    f: &mut impl FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>,
) -> ControlFlow<B> {
    match e.kind {
        ExprKind::Block(Block { expr: Some(e), .. }, _) => for_each_value_source(e, f),
        ExprKind::Match(_, arms, _) => {
            for arm in arms {
                for_each_value_source(arm.body, f)?;
            }
            Continue(())
        },
        ExprKind::If(_, if_expr, Some(else_expr)) => {
            for_each_value_source(if_expr, f)?;
            for_each_value_source(else_expr, f)
        },
        ExprKind::DropTemps(e) => for_each_value_source(e, f),
        _ => f(e),
    }
}

/// Runs the given function for each path expression referencing the given local which occur after
/// the given expression.
pub fn for_each_local_use_after_expr<'tcx, B>(
    cx: &LateContext<'tcx>,
    local_id: HirId,
    expr_id: HirId,
    f: impl FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>,
) -> ControlFlow<B> {
    struct V<'cx, 'tcx, F> {
        cx: &'cx LateContext<'tcx>,
        local_id: HirId,
        expr_id: HirId,
        found: bool,
        f: F,
    }
    impl<'cx, 'tcx, F: FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>, B> Visitor<'tcx> for V<'cx, 'tcx, F> {
        type NestedFilter = nested_filter::OnlyBodies;
        type BreakTy = B;
        fn nested_visit_map(&mut self) -> Self::Map {
            self.cx.tcx.hir()
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'tcx>) -> ControlFlow<B> {
            if !self.found {
                if e.hir_id == self.expr_id {
                    self.found = true;
                    Continue(())
                } else {
                    walk_expr(self, e)
                }
            } else if path_to_local_id(e, self.local_id) {
                (self.f)(e)
            } else {
                walk_expr(self, e)
            }
        }
    }

    if let Some(b) = get_enclosing_block(cx, local_id) {
        let mut v = V {
            cx,
            local_id,
            expr_id,
            found: false,
            f,
        };
        v.visit_block(b)
    } else {
        Continue(())
    }
}

// Calls the given function for every unconsumed temporary created by the expression. Note the
// function is only guaranteed to be called for types which need to be dropped, but it may be called
// for other types.
#[allow(clippy::too_many_lines)]
pub fn for_each_unconsumed_temporary<'tcx, B>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'tcx>,
    mut f: impl FnMut(Ty<'tcx>) -> ControlFlow<B>,
) -> ControlFlow<B> {
    // Todo: Handle partially consumed values.
    fn helper<'tcx, B>(
        typeck: &'tcx TypeckResults<'tcx>,
        consume: bool,
        e: &'tcx Expr<'tcx>,
        f: &mut impl FnMut(Ty<'tcx>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        if !consume
            || matches!(
                typeck.expr_adjustments(e),
                [adjust, ..] if matches!(adjust.kind, Adjust::Borrow(_) | Adjust::Deref(_))
            )
        {
            match e.kind {
                ExprKind::Path(QPath::Resolved(None, p))
                    if matches!(p.res, Res::Def(DefKind::Ctor(_, CtorKind::Const), _)) =>
                {
                    f(typeck.expr_ty(e))?;
                },
                ExprKind::Path(_)
                | ExprKind::Unary(UnOp::Deref, _)
                | ExprKind::Index(..)
                | ExprKind::Field(..)
                | ExprKind::AddrOf(..) => (),
                _ => f(typeck.expr_ty(e))?,
            }
        }
        match e.kind {
            ExprKind::AddrOf(_, _, e)
            | ExprKind::Field(e, _)
            | ExprKind::Unary(UnOp::Deref, e)
            | ExprKind::Match(e, ..)
            | ExprKind::Let(&Let { init: e, .. }) => {
                helper(typeck, false, e, f)?;
            },
            ExprKind::Block(&Block { expr: Some(e), .. }, _)
            | ExprKind::Box(e)
            | ExprKind::Cast(e, _)
            | ExprKind::Unary(_, e) => {
                helper(typeck, true, e, f)?;
            },
            ExprKind::Call(callee, args) => {
                helper(typeck, true, callee, f)?;
                for arg in args {
                    helper(typeck, true, arg, f)?;
                }
            },
            ExprKind::MethodCall(_, receiver, args, _) => {
                helper(typeck, true, receiver, f)?;
                for arg in args {
                    helper(typeck, true, arg, f)?;
                }
            },
            ExprKind::Tup(args) | ExprKind::Array(args) => {
                for arg in args {
                    helper(typeck, true, arg, f)?;
                }
            },
            ExprKind::Index(borrowed, consumed)
            | ExprKind::Assign(borrowed, consumed, _)
            | ExprKind::AssignOp(_, borrowed, consumed) => {
                helper(typeck, false, borrowed, f)?;
                helper(typeck, true, consumed, f)?;
            },
            ExprKind::Binary(_, lhs, rhs) => {
                helper(typeck, true, lhs, f)?;
                helper(typeck, true, rhs, f)?;
            },
            ExprKind::Struct(_, fields, default) => {
                for field in fields {
                    helper(typeck, true, field.expr, f)?;
                }
                if let Some(default) = default {
                    helper(typeck, false, default, f)?;
                }
            },
            ExprKind::If(cond, then, else_expr) => {
                helper(typeck, true, cond, f)?;
                helper(typeck, true, then, f)?;
                if let Some(else_expr) = else_expr {
                    helper(typeck, true, else_expr, f)?;
                }
            },
            ExprKind::Type(e, _) => {
                helper(typeck, consume, e, f)?;
            },

            // Either drops temporaries, jumps out of the current expression, or has no sub expression.
            ExprKind::DropTemps(_)
            | ExprKind::Ret(_)
            | ExprKind::Break(..)
            | ExprKind::Yield(..)
            | ExprKind::Block(..)
            | ExprKind::Loop(..)
            | ExprKind::Repeat(..)
            | ExprKind::Lit(_)
            | ExprKind::ConstBlock(_)
            | ExprKind::Closure { .. }
            | ExprKind::Path(_)
            | ExprKind::Continue(_)
            | ExprKind::InlineAsm(_)
            | ExprKind::Err(_) => (),
        }
        Continue(())
    }
    helper(cx.typeck_results(), true, e, &mut f)
}

pub fn any_temporaries_need_ordered_drop<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) -> bool {
    for_each_unconsumed_temporary(cx, e, |ty| {
        if needs_ordered_drop(cx, ty) {
            Break(())
        } else {
            Continue(())
        }
    })
    .is_break()
}

/// Runs the given function for each path expression referencing the given local which occur after
/// the given expression.
pub fn for_each_local_assignment<'tcx, B>(
    cx: &LateContext<'tcx>,
    local_id: HirId,
    f: impl FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>,
) -> ControlFlow<B> {
    struct V<'cx, 'tcx, F> {
        cx: &'cx LateContext<'tcx>,
        local_id: HirId,
        f: F,
    }
    impl<'cx, 'tcx, F: FnMut(&'tcx Expr<'tcx>) -> ControlFlow<B>, B> Visitor<'tcx> for V<'cx, 'tcx, F> {
        type NestedFilter = nested_filter::OnlyBodies;
        type BreakTy = B;
        fn nested_visit_map(&mut self) -> Self::Map {
            self.cx.tcx.hir()
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'tcx>) -> ControlFlow<B> {
            if let ExprKind::Assign(lhs, rhs, _) = e.kind
                && path_to_local_id(lhs, self.local_id)
            {
                (self.f)(rhs)?;
                self.visit_expr(rhs)
            } else {
                walk_expr(self, e)
            }
        }
    }

    if let Some(b) = get_enclosing_block(cx, local_id) {
        let mut v = V { cx, local_id, f };
        v.visit_block(b)
    } else {
        Continue(())
    }
}

pub fn contains_break_or_continue(expr: &Expr<'_>) -> bool {
    for_each_expr(expr, |e| {
        if matches!(e.kind, ExprKind::Break(..) | ExprKind::Continue(..)) {
            Break(())
        } else {
            Continue(())
        }
    })
    .is_some()
}
