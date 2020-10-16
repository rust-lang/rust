//! Checks for uses of const which the type is not `Freeze` (`Cell`-free).
//!
//! This lint is **warn** by default.

use std::ptr;

use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, ImplItem, ImplItemKind, Item, ItemKind, Node, TraitItem, TraitItemKind, UnOp};
use rustc_infer::traits::specialization_graph;
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{AssocKind, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{InnerSpan, Span, DUMMY_SP};
use rustc_typeck::hir_ty_to_ty;

use crate::utils::{in_constant, qpath_res, span_lint_and_then};
use if_chain::if_chain;

// FIXME: this is a correctness problem but there's no suitable
// warn-by-default category.
declare_clippy_lint! {
    /// **What it does:** Checks for declaration of `const` items which is interior
    /// mutable (e.g., contains a `Cell`, `Mutex`, `AtomicXxxx`, etc.).
    ///
    /// **Why is this bad?** Consts are copied everywhere they are referenced, i.e.,
    /// every time you refer to the const a fresh instance of the `Cell` or `Mutex`
    /// or `AtomicXxxx` will be created, which defeats the whole purpose of using
    /// these types in the first place.
    ///
    /// The `const` should better be replaced by a `static` item if a global
    /// variable is wanted, or replaced by a `const fn` if a constructor is wanted.
    ///
    /// **Known problems:** A "non-constant" const item is a legacy way to supply an
    /// initialized value to downstream `static` items (e.g., the
    /// `std::sync::ONCE_INIT` constant). In this case the use of `const` is legit,
    /// and this lint should be suppressed.
    ///
    /// When an enum has variants with interior mutability, use of its non interior mutable
    /// variants can generate false positives. See issue
    /// [#3962](https://github.com/rust-lang/rust-clippy/issues/3962)
    ///
    /// Types that have underlying or potential interior mutability trigger the lint whether
    /// the interior mutable field is used or not. See issues
    /// [#5812](https://github.com/rust-lang/rust-clippy/issues/5812) and
    /// [#3825](https://github.com/rust-lang/rust-clippy/issues/3825)
    ///
    /// **Example:**
    /// ```rust
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    ///
    /// // Bad.
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    ///
    /// // Good.
    /// static STATIC_ATOM: AtomicUsize = AtomicUsize::new(15);
    /// STATIC_ATOM.store(9, SeqCst);
    /// assert_eq!(STATIC_ATOM.load(SeqCst), 9); // use a `static` item to refer to the same instance
    /// ```
    pub DECLARE_INTERIOR_MUTABLE_CONST,
    style,
    "declaring `const` with interior mutability"
}

// FIXME: this is a correctness problem but there's no suitable
// warn-by-default category.
declare_clippy_lint! {
    /// **What it does:** Checks if `const` items which is interior mutable (e.g.,
    /// contains a `Cell`, `Mutex`, `AtomicXxxx`, etc.) has been borrowed directly.
    ///
    /// **Why is this bad?** Consts are copied everywhere they are referenced, i.e.,
    /// every time you refer to the const a fresh instance of the `Cell` or `Mutex`
    /// or `AtomicXxxx` will be created, which defeats the whole purpose of using
    /// these types in the first place.
    ///
    /// The `const` value should be stored inside a `static` item.
    ///
    /// **Known problems:** When an enum has variants with interior mutability, use of its non
    /// interior mutable variants can generate false positives. See issue
    /// [#3962](https://github.com/rust-lang/rust-clippy/issues/3962)
    ///
    /// Types that have underlying or potential interior mutability trigger the lint whether
    /// the interior mutable field is used or not. See issues
    /// [#5812](https://github.com/rust-lang/rust-clippy/issues/5812) and
    /// [#3825](https://github.com/rust-lang/rust-clippy/issues/3825)
    ///
    /// **Example:**
    /// ```rust
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    ///
    /// // Bad.
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    ///
    /// // Good.
    /// static STATIC_ATOM: AtomicUsize = CONST_ATOM;
    /// STATIC_ATOM.store(9, SeqCst);
    /// assert_eq!(STATIC_ATOM.load(SeqCst), 9); // use a `static` item to refer to the same instance
    /// ```
    pub BORROW_INTERIOR_MUTABLE_CONST,
    style,
    "referencing `const` with interior mutability"
}

#[derive(Copy, Clone)]
enum Source {
    Item { item: Span },
    Assoc { item: Span },
    Expr { expr: Span },
}

impl Source {
    #[must_use]
    fn lint(&self) -> (&'static Lint, &'static str, Span) {
        match self {
            Self::Item { item } | Self::Assoc { item, .. } => (
                DECLARE_INTERIOR_MUTABLE_CONST,
                "a `const` item should never be interior mutable",
                *item,
            ),
            Self::Expr { expr } => (
                BORROW_INTERIOR_MUTABLE_CONST,
                "a `const` item with interior mutability should not be borrowed",
                *expr,
            ),
        }
    }
}

fn verify_ty_bound<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, source: Source) {
    // Ignore types whose layout is unknown since `is_freeze` reports every generic types as `!Freeze`,
    // making it indistinguishable from `UnsafeCell`. i.e. it isn't a tool to prove a type is
    // 'unfrozen'. However, this code causes a false negative in which
    // a type contains a layout-unknown type, but also a unsafe cell like `const CELL: Cell<T>`.
    // Yet, it's better than `ty.has_type_flags(TypeFlags::HAS_TY_PARAM | TypeFlags::HAS_PROJECTION)`
    // since it works when a pointer indirection involves (`Cell<*const T>`).
    // Making up a `ParamEnv` where every generic params and assoc types are `Freeze`is another option;
    // but I'm not sure whether it's a decent way, if possible.
    if cx.tcx.layout_of(cx.param_env.and(ty)).is_err() || ty.is_freeze(cx.tcx.at(DUMMY_SP), cx.param_env) {
        return;
    }

    let (lint, msg, span) = source.lint();
    span_lint_and_then(cx, lint, span, msg, |diag| {
        if span.from_expansion() {
            return; // Don't give suggestions into macros.
        }
        match source {
            Source::Item { .. } => {
                let const_kw_span = span.from_inner(InnerSpan::new(0, 5));
                diag.span_label(const_kw_span, "make this a static item (maybe with lazy_static)");
            },
            Source::Assoc { .. } => (),
            Source::Expr { .. } => {
                diag.help("assign this const to a local or static variable, and use the variable here");
            },
        }
    });
}

declare_lint_pass!(NonCopyConst => [DECLARE_INTERIOR_MUTABLE_CONST, BORROW_INTERIOR_MUTABLE_CONST]);

impl<'tcx> LateLintPass<'tcx> for NonCopyConst {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx Item<'_>) {
        if let ItemKind::Const(hir_ty, ..) = &it.kind {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            verify_ty_bound(cx, ty, Source::Item { item: it.span });
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Const(hir_ty, ..) = &trait_item.kind {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            // Normalize assoc types because ones originated from generic params
            // bounded other traits could have their bound.
            let normalized = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
            verify_ty_bound(cx, normalized, Source::Assoc { item: trait_item.span });
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Const(hir_ty, ..) = &impl_item.kind {
            let item_hir_id = cx.tcx.hir().get_parent_node(impl_item.hir_id);
            let item = cx.tcx.hir().expect_item(item_hir_id);

            match &item.kind {
                ItemKind::Impl {
                    of_trait: Some(of_trait_ref),
                    ..
                } => {
                    if_chain! {
                        // Lint a trait impl item only when the definition is a generic type,
                        // assuming a assoc const is not meant to be a interior mutable type.
                        if let Some(of_trait_def_id) = of_trait_ref.trait_def_id();
                        if let Some(of_assoc_item) = specialization_graph::Node::Trait(of_trait_def_id)
                            .item(cx.tcx, impl_item.ident, AssocKind::Const, of_trait_def_id);
                        if cx
                            .tcx
                            .layout_of(cx.tcx.param_env(of_trait_def_id).and(
                                // Normalize assoc types because ones originated from generic params
                                // bounded other traits could have their bound at the trait defs;
                                // and, in that case, the definition is *not* generic.
                                cx.tcx.normalize_erasing_regions(
                                    cx.tcx.param_env(of_trait_def_id),
                                    cx.tcx.type_of(of_assoc_item.def_id),
                                ),
                            ))
                            .is_err();
                        then {
                            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
                            let normalized = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
                            verify_ty_bound(
                                cx,
                                normalized,
                                Source::Assoc {
                                    item: impl_item.span,
                                },
                            );
                        }
                    }
                },
                ItemKind::Impl { of_trait: None, .. } => {
                    let ty = hir_ty_to_ty(cx.tcx, hir_ty);
                    // Normalize assoc types originated from generic params.
                    let normalized = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
                    verify_ty_bound(cx, normalized, Source::Assoc { item: impl_item.span });
                },
                _ => (),
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Path(qpath) = &expr.kind {
            // Only lint if we use the const item inside a function.
            if in_constant(cx, expr.hir_id) {
                return;
            }

            // Make sure it is a const item.
            match qpath_res(cx, qpath, expr.hir_id) {
                Res::Def(DefKind::Const | DefKind::AssocConst, _) => {},
                _ => return,
            };

            // Climb up to resolve any field access and explicit referencing.
            let mut cur_expr = expr;
            let mut dereferenced_expr = expr;
            let mut needs_check_adjustment = true;
            loop {
                let parent_id = cx.tcx.hir().get_parent_node(cur_expr.hir_id);
                if parent_id == cur_expr.hir_id {
                    break;
                }
                if let Some(Node::Expr(parent_expr)) = cx.tcx.hir().find(parent_id) {
                    match &parent_expr.kind {
                        ExprKind::AddrOf(..) => {
                            // `&e` => `e` must be referenced.
                            needs_check_adjustment = false;
                        },
                        ExprKind::Field(..) => {
                            needs_check_adjustment = true;

                            // Check whether implicit dereferences happened;
                            // if so, no need to go further up
                            // because of the same reason as the `ExprKind::Unary` case.
                            if cx
                                .typeck_results()
                                .expr_adjustments(dereferenced_expr)
                                .iter()
                                .any(|adj| matches!(adj.kind, Adjust::Deref(_)))
                            {
                                break;
                            }

                            dereferenced_expr = parent_expr;
                        },
                        ExprKind::Index(e, _) if ptr::eq(&**e, cur_expr) => {
                            // `e[i]` => desugared to `*Index::index(&e, i)`,
                            // meaning `e` must be referenced.
                            // no need to go further up since a method call is involved now.
                            needs_check_adjustment = false;
                            break;
                        },
                        ExprKind::Unary(UnOp::UnDeref, _) => {
                            // `*e` => desugared to `*Deref::deref(&e)`,
                            // meaning `e` must be referenced.
                            // no need to go further up since a method call is involved now.
                            needs_check_adjustment = false;
                            break;
                        },
                        _ => break,
                    }
                    cur_expr = parent_expr;
                } else {
                    break;
                }
            }

            let ty = if needs_check_adjustment {
                let adjustments = cx.typeck_results().expr_adjustments(dereferenced_expr);
                if let Some(i) = adjustments
                    .iter()
                    .position(|adj| matches!(adj.kind, Adjust::Borrow(_) | Adjust::Deref(_)))
                {
                    if i == 0 {
                        cx.typeck_results().expr_ty(dereferenced_expr)
                    } else {
                        adjustments[i - 1].target
                    }
                } else {
                    // No borrow adjustments means the entire const is moved.
                    return;
                }
            } else {
                cx.typeck_results().expr_ty(dereferenced_expr)
            };

            verify_ty_bound(cx, ty, Source::Expr { expr: expr.span });
        }
    }
}
