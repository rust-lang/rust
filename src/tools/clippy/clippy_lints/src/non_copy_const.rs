//! Checks for usage of const which the type is not `Freeze` (`Cell`-free).
//!
//! This lint is **warn** by default.

use std::ptr;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::in_constant;
use clippy_utils::macros::macro_backtrace;
use if_chain::if_chain;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{
    BodyId, Expr, ExprKind, HirId, Impl, ImplItem, ImplItemKind, Item, ItemKind, Node, TraitItem, TraitItemKind, UnOp,
};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, InnerSpan, Span};
use rustc_target::abi::VariantIdx;
use rustc_middle::mir::interpret::EvalToValTreeResult;
use rustc_middle::mir::interpret::GlobalId;

// FIXME: this is a correctness problem but there's no suitable
// warn-by-default category.
declare_clippy_lint! {
    /// ### What it does
    /// Checks for declaration of `const` items which is interior
    /// mutable (e.g., contains a `Cell`, `Mutex`, `AtomicXxxx`, etc.).
    ///
    /// ### Why is this bad?
    /// Consts are copied everywhere they are referenced, i.e.,
    /// every time you refer to the const a fresh instance of the `Cell` or `Mutex`
    /// or `AtomicXxxx` will be created, which defeats the whole purpose of using
    /// these types in the first place.
    ///
    /// The `const` should better be replaced by a `static` item if a global
    /// variable is wanted, or replaced by a `const fn` if a constructor is wanted.
    ///
    /// ### Known problems
    /// A "non-constant" const item is a legacy way to supply an
    /// initialized value to downstream `static` items (e.g., the
    /// `std::sync::ONCE_INIT` constant). In this case the use of `const` is legit,
    /// and this lint should be suppressed.
    ///
    /// Even though the lint avoids triggering on a constant whose type has enums that have variants
    /// with interior mutability, and its value uses non interior mutable variants (see
    /// [#3962](https://github.com/rust-lang/rust-clippy/issues/3962) and
    /// [#3825](https://github.com/rust-lang/rust-clippy/issues/3825) for examples);
    /// it complains about associated constants without default values only based on its types;
    /// which might not be preferable.
    /// There're other enums plus associated constants cases that the lint cannot handle.
    ///
    /// Types that have underlying or potential interior mutability trigger the lint whether
    /// the interior mutable field is used or not. See issues
    /// [#5812](https://github.com/rust-lang/rust-clippy/issues/5812) and
    ///
    /// ### Example
    /// ```rust
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    ///
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// static STATIC_ATOM: AtomicUsize = AtomicUsize::new(15);
    /// STATIC_ATOM.store(9, SeqCst);
    /// assert_eq!(STATIC_ATOM.load(SeqCst), 9); // use a `static` item to refer to the same instance
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DECLARE_INTERIOR_MUTABLE_CONST,
    style,
    "declaring `const` with interior mutability"
}

// FIXME: this is a correctness problem but there's no suitable
// warn-by-default category.
declare_clippy_lint! {
    /// ### What it does
    /// Checks if `const` items which is interior mutable (e.g.,
    /// contains a `Cell`, `Mutex`, `AtomicXxxx`, etc.) has been borrowed directly.
    ///
    /// ### Why is this bad?
    /// Consts are copied everywhere they are referenced, i.e.,
    /// every time you refer to the const a fresh instance of the `Cell` or `Mutex`
    /// or `AtomicXxxx` will be created, which defeats the whole purpose of using
    /// these types in the first place.
    ///
    /// The `const` value should be stored inside a `static` item.
    ///
    /// ### Known problems
    /// When an enum has variants with interior mutability, use of its non
    /// interior mutable variants can generate false positives. See issue
    /// [#3962](https://github.com/rust-lang/rust-clippy/issues/3962)
    ///
    /// Types that have underlying or potential interior mutability trigger the lint whether
    /// the interior mutable field is used or not. See issues
    /// [#5812](https://github.com/rust-lang/rust-clippy/issues/5812) and
    /// [#3825](https://github.com/rust-lang/rust-clippy/issues/3825)
    ///
    /// ### Example
    /// ```rust
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    ///
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    ///
    /// static STATIC_ATOM: AtomicUsize = CONST_ATOM;
    /// STATIC_ATOM.store(9, SeqCst);
    /// assert_eq!(STATIC_ATOM.load(SeqCst), 9); // use a `static` item to refer to the same instance
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BORROW_INTERIOR_MUTABLE_CONST,
    style,
    "referencing `const` with interior mutability"
}

fn is_unfrozen<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    // Ignore types whose layout is unknown since `is_freeze` reports every generic types as `!Freeze`,
    // making it indistinguishable from `UnsafeCell`. i.e. it isn't a tool to prove a type is
    // 'unfrozen'. However, this code causes a false negative in which
    // a type contains a layout-unknown type, but also an unsafe cell like `const CELL: Cell<T>`.
    // Yet, it's better than `ty.has_type_flags(TypeFlags::HAS_TY_PARAM | TypeFlags::HAS_PROJECTION)`
    // since it works when a pointer indirection involves (`Cell<*const T>`).
    // Making up a `ParamEnv` where every generic params and assoc types are `Freeze`is another option;
    // but I'm not sure whether it's a decent way, if possible.
    cx.tcx.layout_of(cx.param_env.and(ty)).is_ok() && !ty.is_freeze(cx.tcx, cx.param_env)
}

fn is_value_unfrozen_raw<'tcx>(
    cx: &LateContext<'tcx>,
    result: Result<Option<ty::ValTree<'tcx>>, ErrorHandled>,
    ty: Ty<'tcx>,
) -> bool {
    fn inner<'tcx>(cx: &LateContext<'tcx>, val: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> bool {
        match *ty.kind() {
            // the fact that we have to dig into every structs to search enums
            // leads us to the point checking `UnsafeCell` directly is the only option.
            ty::Adt(ty_def, ..) if ty_def.is_unsafe_cell() => true,
            // As of 2022-09-08 miri doesn't track which union field is active so there's no safe way to check the
            // contained value.
            ty::Adt(def, ..) if def.is_union() => false,
            ty::Array(ty, _)  => {
                val.unwrap_branch().iter().any(|field| inner(cx, *field, ty))
            },
            ty::Adt(def, _) if def.is_union() => false,
            ty::Adt(def, substs) if def.is_enum() => {
                let (&variant_index, fields) = val.unwrap_branch().split_first().unwrap();
                let variant_index =
                    VariantIdx::from_u32(variant_index.unwrap_leaf().try_to_u32().ok().unwrap());
                fields.iter().copied().zip(
                    def.variants()[variant_index]
                        .fields
                        .iter()
                        .map(|field| field.ty(cx.tcx, substs))).any(|(field, ty)| inner(cx, field, ty))
            }
            ty::Adt(def, substs) => {
                val.unwrap_branch().iter().zip(def.non_enum_variant().fields.iter().map(|field| field.ty(cx.tcx, substs))).any(|(field, ty)| inner(cx, *field, ty))
            }
            ty::Tuple(tys) => val.unwrap_branch().iter().zip(tys).any(|(field, ty)| inner(cx, *field, ty)),
            _ => false,
        }
    }
    result.map_or_else(
        |err| {
            // Consider `TooGeneric` cases as being unfrozen.
            // This causes a false positive where an assoc const whose type is unfrozen
            // have a value that is a frozen variant with a generic param (an example is
            // `declare_interior_mutable_const::enums::BothOfCellAndGeneric::GENERIC_VARIANT`).
            // However, it prevents a number of false negatives that is, I think, important:
            // 1. assoc consts in trait defs referring to consts of themselves (an example is
            //    `declare_interior_mutable_const::traits::ConcreteTypes::ANOTHER_ATOMIC`).
            // 2. a path expr referring to assoc consts whose type is doesn't have any frozen variants in trait
            //    defs (i.e. without substitute for `Self`). (e.g. borrowing
            //    `borrow_interior_mutable_const::trait::ConcreteTypes::ATOMIC`)
            // 3. similar to the false positive above; but the value is an unfrozen variant, or the type has no
            //    enums. (An example is
            //    `declare_interior_mutable_const::enums::BothOfCellAndGeneric::UNFROZEN_VARIANT` and
            //    `declare_interior_mutable_const::enums::BothOfCellAndGeneric::NO_ENUM`).
            // One might be able to prevent these FNs correctly, and replace this with `false`;
            // e.g. implementing `has_frozen_variant` described above, and not running this function
            // when the type doesn't have any frozen variants would be the 'correct' way for the 2nd
            // case (that actually removes another suboptimal behavior (I won't say 'false positive') where,
            // similar to 2., but with the a frozen variant) (e.g. borrowing
            // `borrow_interior_mutable_const::enums::AssocConsts::TO_BE_FROZEN_VARIANT`).
            // I chose this way because unfrozen enums as assoc consts are rare (or, hopefully, none).
            err == ErrorHandled::TooGeneric
        },
        |val| val.map_or(true, |val| inner(cx, val, ty)),
    )
}

fn is_value_unfrozen_poly<'tcx>(cx: &LateContext<'tcx>, body_id: BodyId, ty: Ty<'tcx>) -> bool {
    let def_id = body_id.hir_id.owner.to_def_id();
    let substs = ty::InternalSubsts::identity_for_item(cx.tcx, def_id);
    let instance = ty::Instance::new(def_id, substs);
    let cid = rustc_middle::mir::interpret::GlobalId { instance, promoted: None };
    let param_env = cx.tcx.param_env(def_id).with_reveal_all_normalized(cx.tcx);
    let result = cx.tcx.const_eval_global_id_for_typeck(param_env, cid, None);
    is_value_unfrozen_raw(cx, result, ty)
}

fn is_value_unfrozen_expr<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId, def_id: DefId, ty: Ty<'tcx>) -> bool {
    let substs = cx.typeck_results().node_substs(hir_id);

    let result = const_eval_resolve(cx.tcx, cx.param_env, ty::UnevaluatedConst::new(def_id, substs), None);
    is_value_unfrozen_raw(cx, result, ty)
}


pub fn const_eval_resolve<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ct: ty::UnevaluatedConst<'tcx>,
    span: Option<Span>,
) -> EvalToValTreeResult<'tcx> {
    match ty::Instance::resolve(tcx, param_env, ct.def, ct.substs) {
        Ok(Some(instance)) => {
            let cid = GlobalId { instance, promoted: None };
            tcx.const_eval_global_id_for_typeck(param_env, cid, span)
        }
        Ok(None) => Err(ErrorHandled::TooGeneric),
        Err(err) => Err(ErrorHandled::Reported(err.into())),
    }
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

fn lint(cx: &LateContext<'_>, source: Source) {
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
        if let ItemKind::Const(hir_ty, body_id) = it.kind {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            if !ignored_macro(cx, it) && is_unfrozen(cx, ty) && is_value_unfrozen_poly(cx, body_id, ty) {
                lint(cx, Source::Item { item: it.span });
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Const(hir_ty, body_id_opt) = &trait_item.kind {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);

            // Normalize assoc types because ones originated from generic params
            // bounded other traits could have their bound.
            let normalized = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
            if is_unfrozen(cx, normalized)
                // When there's no default value, lint it only according to its type;
                // in other words, lint consts whose value *could* be unfrozen, not definitely is.
                // This feels inconsistent with how the lint treats generic types,
                // which avoids linting types which potentially become unfrozen.
                // One could check whether an unfrozen type have a *frozen variant*
                // (like `body_id_opt.map_or_else(|| !has_frozen_variant(...), ...)`),
                // and do the same as the case of generic types at impl items.
                // Note that it isn't sufficient to check if it has an enum
                // since all of that enum's variants can be unfrozen:
                // i.e. having an enum doesn't necessary mean a type has a frozen variant.
                // And, implementing it isn't a trivial task; it'll probably end up
                // re-implementing the trait predicate evaluation specific to `Freeze`.
                && body_id_opt.map_or(true, |body_id| is_value_unfrozen_poly(cx, body_id, normalized))
            {
                lint(cx, Source::Assoc { item: trait_item.span });
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Const(hir_ty, body_id) = &impl_item.kind {
            let item_def_id = cx.tcx.hir().get_parent_item(impl_item.hir_id()).def_id;
            let item = cx.tcx.hir().expect_item(item_def_id);

            match &item.kind {
                ItemKind::Impl(Impl {
                    of_trait: Some(of_trait_ref),
                    ..
                }) => {
                    if_chain! {
                        // Lint a trait impl item only when the definition is a generic type,
                        // assuming an assoc const is not meant to be an interior mutable type.
                        if let Some(of_trait_def_id) = of_trait_ref.trait_def_id();
                        if let Some(of_assoc_item) = cx
                            .tcx
                            .associated_item(impl_item.owner_id)
                            .trait_item_def_id;
                        if cx
                            .tcx
                            .layout_of(cx.tcx.param_env(of_trait_def_id).and(
                                // Normalize assoc types because ones originated from generic params
                                // bounded other traits could have their bound at the trait defs;
                                // and, in that case, the definition is *not* generic.
                                cx.tcx.normalize_erasing_regions(
                                    cx.tcx.param_env(of_trait_def_id),
                                    cx.tcx.type_of(of_assoc_item).subst_identity(),
                                ),
                            ))
                            .is_err();
                            // If there were a function like `has_frozen_variant` described above,
                            // we should use here as a frozen variant is a potential to be frozen
                            // similar to unknown layouts.
                            // e.g. `layout_of(...).is_err() || has_frozen_variant(...);`
                        let ty = hir_ty_to_ty(cx.tcx, hir_ty);
                        let normalized = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
                        if is_unfrozen(cx, normalized);
                        if is_value_unfrozen_poly(cx, *body_id, normalized);
                        then {
                            lint(
                               cx,
                               Source::Assoc {
                                   item: impl_item.span,
                                },
                            );
                        }
                    }
                },
                ItemKind::Impl(Impl { of_trait: None, .. }) => {
                    let ty = hir_ty_to_ty(cx.tcx, hir_ty);
                    // Normalize assoc types originated from generic params.
                    let normalized = cx.tcx.normalize_erasing_regions(cx.param_env, ty);

                    if is_unfrozen(cx, ty) && is_value_unfrozen_poly(cx, *body_id, normalized) {
                        lint(cx, Source::Assoc { item: impl_item.span });
                    }
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
            let Res::Def(DefKind::Const | DefKind::AssocConst, item_def_id) = cx.qpath_res(qpath, expr.hir_id) else {
                return
            };

            // Climb up to resolve any field access and explicit referencing.
            let mut cur_expr = expr;
            let mut dereferenced_expr = expr;
            let mut needs_check_adjustment = true;
            loop {
                let parent_id = cx.tcx.hir().parent_id(cur_expr.hir_id);
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
                        ExprKind::Unary(UnOp::Deref, _) => {
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

            if is_unfrozen(cx, ty) && is_value_unfrozen_expr(cx, expr.hir_id, item_def_id, ty) {
                lint(cx, Source::Expr { expr: expr.span });
            }
        }
    }
}

fn ignored_macro(cx: &LateContext<'_>, it: &rustc_hir::Item<'_>) -> bool {
    macro_backtrace(it.span).any(|macro_call| {
        matches!(
            cx.tcx.get_diagnostic_name(macro_call.def_id),
            Some(sym::thread_local_macro)
        )
    })
}
