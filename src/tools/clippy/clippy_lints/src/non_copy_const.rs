use std::ptr;

use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_const_context;
use clippy_utils::macros::macro_backtrace;
use clippy_utils::ty::{InteriorMut, implements_trait};
use rustc_abi::VariantIdx;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{
    BodyId, Expr, ExprKind, HirId, Impl, ImplItem, ImplItemKind, Item, ItemKind, Node, TraitItem, TraitItemKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_middle::mir::interpret::{ErrorHandled, EvalToValTreeResult, GlobalId, ReportedErrorInfo};
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::{DUMMY_SP, Span, sym};

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
    /// the interior mutable field is used or not. See issue
    /// [#5812](https://github.com/rust-lang/rust-clippy/issues/5812)
    ///
    /// ### Example
    /// ```no_run
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    ///
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ### Example
    /// ```no_run
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    ///
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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

#[derive(Copy, Clone)]
enum Source<'tcx> {
    Item { item: Span, ty: Ty<'tcx> },
    Assoc { item: Span },
    Expr { expr: Span },
}

impl Source<'_> {
    #[must_use]
    fn lint(&self) -> (&'static Lint, &'static str, Span) {
        match self {
            Self::Item { item, .. } | Self::Assoc { item, .. } => (
                DECLARE_INTERIOR_MUTABLE_CONST,
                "a `const` item should not be interior mutable",
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

fn lint<'tcx>(cx: &LateContext<'tcx>, source: Source<'tcx>) {
    let (lint, msg, span) = source.lint();
    span_lint_and_then(cx, lint, span, msg, |diag| {
        if span.from_expansion() {
            return; // Don't give suggestions into macros.
        }
        match source {
            Source::Item { ty, .. } => {
                let Some(sync_trait) = cx.tcx.lang_items().sync_trait() else {
                    return;
                };
                if implements_trait(cx, ty, sync_trait, &[]) {
                    diag.help("consider making this a static item");
                } else {
                    diag.help(
                        "consider making this `Sync` so that it can go in a static item or using a `thread_local`",
                    );
                }
            },
            Source::Assoc { .. } => (),
            Source::Expr { .. } => {
                diag.help("assign this const to a local or static variable, and use the variable here");
            },
        }
    });
}

pub struct NonCopyConst<'tcx> {
    interior_mut: InteriorMut<'tcx>,
}

impl_lint_pass!(NonCopyConst<'_> => [DECLARE_INTERIOR_MUTABLE_CONST, BORROW_INTERIOR_MUTABLE_CONST]);

impl<'tcx> NonCopyConst<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, conf: &'static Conf) -> Self {
        Self {
            interior_mut: InteriorMut::without_pointers(tcx, &conf.ignore_interior_mutability),
        }
    }

    fn is_value_unfrozen_raw_inner(cx: &LateContext<'tcx>, val: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> bool {
        // No branch that we check (yet) should continue if val isn't a branch
        let Some(branched_val) = val.try_to_branch() else {
            return false;
        };
        match *ty.kind() {
            // the fact that we have to dig into every structs to search enums
            // leads us to the point checking `UnsafeCell` directly is the only option.
            ty::Adt(ty_def, ..) if ty_def.is_unsafe_cell() => true,
            // As of 2022-09-08 miri doesn't track which union field is active so there's no safe way to check the
            // contained value.
            ty::Adt(def, ..) if def.is_union() => false,
            ty::Array(ty, _) => branched_val
                .iter()
                .any(|field| Self::is_value_unfrozen_raw_inner(cx, *field, ty)),
            ty::Adt(def, args) if def.is_enum() => {
                let Some((&variant_valtree, fields)) = branched_val.split_first() else {
                    return false;
                };
                let variant_index = variant_valtree.unwrap_leaf();
                let variant_index = VariantIdx::from_u32(variant_index.to_u32());
                fields
                    .iter()
                    .copied()
                    .zip(
                        def.variants()[variant_index]
                            .fields
                            .iter()
                            .map(|field| field.ty(cx.tcx, args)),
                    )
                    .any(|(field, ty)| Self::is_value_unfrozen_raw_inner(cx, field, ty))
            },
            ty::Adt(def, args) => branched_val
                .iter()
                .zip(def.non_enum_variant().fields.iter().map(|field| field.ty(cx.tcx, args)))
                .any(|(field, ty)| Self::is_value_unfrozen_raw_inner(cx, *field, ty)),
            ty::Tuple(tys) => branched_val
                .iter()
                .zip(tys)
                .any(|(field, ty)| Self::is_value_unfrozen_raw_inner(cx, *field, ty)),
            ty::Alias(ty::Projection, _) => match cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty) {
                Ok(normalized_ty) if ty != normalized_ty => Self::is_value_unfrozen_raw_inner(cx, val, normalized_ty),
                _ => false,
            },
            _ => false,
        }
    }

    fn is_value_unfrozen_raw(
        cx: &LateContext<'tcx>,
        result: Result<Result<ty::ValTree<'tcx>, Ty<'tcx>>, ErrorHandled>,
        ty: Ty<'tcx>,
    ) -> bool {
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
                // similar to 2., but with a frozen variant) (e.g. borrowing
                // `borrow_interior_mutable_const::enums::AssocConsts::TO_BE_FROZEN_VARIANT`).
                // I chose this way because unfrozen enums as assoc consts are rare (or, hopefully, none).
                matches!(err, ErrorHandled::TooGeneric(..))
            },
            |val| val.map_or(true, |val| Self::is_value_unfrozen_raw_inner(cx, val, ty)),
        )
    }

    fn is_value_unfrozen_poly(cx: &LateContext<'tcx>, body_id: BodyId, ty: Ty<'tcx>) -> bool {
        let def_id = body_id.hir_id.owner.to_def_id();
        let args = ty::GenericArgs::identity_for_item(cx.tcx, def_id);
        let instance = ty::Instance::new(def_id, args);
        let cid = GlobalId {
            instance,
            promoted: None,
        };
        let typing_env = ty::TypingEnv::post_analysis(cx.tcx, def_id);
        let result = cx.tcx.const_eval_global_id_for_typeck(typing_env, cid, DUMMY_SP);
        Self::is_value_unfrozen_raw(cx, result, ty)
    }

    fn is_value_unfrozen_expr(cx: &LateContext<'tcx>, hir_id: HirId, def_id: DefId, ty: Ty<'tcx>) -> bool {
        let args = cx.typeck_results().node_args(hir_id);

        let result = Self::const_eval_resolve(
            cx.tcx,
            cx.typing_env(),
            ty::UnevaluatedConst::new(def_id, args),
            DUMMY_SP,
        );
        Self::is_value_unfrozen_raw(cx, result, ty)
    }

    pub fn const_eval_resolve(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        ct: ty::UnevaluatedConst<'tcx>,
        span: Span,
    ) -> EvalToValTreeResult<'tcx> {
        match ty::Instance::try_resolve(tcx, typing_env, ct.def, ct.args) {
            Ok(Some(instance)) => {
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                tcx.const_eval_global_id_for_typeck(typing_env, cid, span)
            },
            Ok(None) => Err(ErrorHandled::TooGeneric(span)),
            Err(err) => Err(ErrorHandled::Reported(
                ReportedErrorInfo::non_const_eval_error(err),
                span,
            )),
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonCopyConst<'tcx> {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx Item<'_>) {
        if let ItemKind::Const(.., body_id) = it.kind {
            let ty = cx.tcx.type_of(it.owner_id).instantiate_identity();
            if !ignored_macro(cx, it)
                && self.interior_mut.is_interior_mut_ty(cx, ty)
                && Self::is_value_unfrozen_poly(cx, body_id, ty)
            {
                lint(cx, Source::Item { item: it.span, ty });
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Const(_, body_id_opt) = &trait_item.kind {
            let ty = cx.tcx.type_of(trait_item.owner_id).instantiate_identity();

            // Normalize assoc types because ones originated from generic params
            // bounded other traits could have their bound.
            let normalized = cx.tcx.normalize_erasing_regions(cx.typing_env(), ty);
            if self.interior_mut.is_interior_mut_ty(cx, normalized)
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
                && body_id_opt.is_none_or(|body_id| Self::is_value_unfrozen_poly(cx, body_id, normalized))
            {
                lint(cx, Source::Assoc { item: trait_item.span });
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Const(_, body_id) = &impl_item.kind {
            let item_def_id = cx.tcx.hir_get_parent_item(impl_item.hir_id()).def_id;
            let item = cx.tcx.hir_expect_item(item_def_id);

            match &item.kind {
                ItemKind::Impl(Impl {
                    of_trait: Some(of_trait_ref),
                    ..
                }) => {
                    if let Some(of_trait_def_id) = of_trait_ref.trait_def_id()
                        // Lint a trait impl item only when the definition is a generic type,
                        // assuming an assoc const is not meant to be an interior mutable type.
                        && let Some(of_assoc_item) = cx
                            .tcx
                            .associated_item(impl_item.owner_id)
                            .trait_item_def_id
                        && cx
                            .tcx
                            .layout_of(ty::TypingEnv::post_analysis(cx.tcx, of_trait_def_id).as_query_input(
                                // Normalize assoc types because ones originated from generic params
                                // bounded other traits could have their bound at the trait defs;
                                // and, in that case, the definition is *not* generic.
                                cx.tcx.normalize_erasing_regions(
                                    ty::TypingEnv::post_analysis(cx.tcx, of_trait_def_id),
                                    cx.tcx.type_of(of_assoc_item).instantiate_identity(),
                                ),
                            ))
                            .is_err()
                            // If there were a function like `has_frozen_variant` described above,
                            // we should use here as a frozen variant is a potential to be frozen
                            // similar to unknown layouts.
                            // e.g. `layout_of(...).is_err() || has_frozen_variant(...);`
                        && let ty = cx.tcx.type_of(impl_item.owner_id).instantiate_identity()
                        && let normalized = cx.tcx.normalize_erasing_regions(cx.typing_env(), ty)
                        && self.interior_mut.is_interior_mut_ty(cx, normalized)
                        && Self::is_value_unfrozen_poly(cx, *body_id, normalized)
                    {
                        lint(cx, Source::Assoc { item: impl_item.span });
                    }
                },
                ItemKind::Impl(Impl { of_trait: None, .. }) => {
                    let ty = cx.tcx.type_of(impl_item.owner_id).instantiate_identity();
                    // Normalize assoc types originated from generic params.
                    let normalized = cx.tcx.normalize_erasing_regions(cx.typing_env(), ty);

                    if self.interior_mut.is_interior_mut_ty(cx, normalized)
                        && Self::is_value_unfrozen_poly(cx, *body_id, normalized)
                    {
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
            if is_in_const_context(cx) {
                return;
            }

            // Make sure it is a const item.
            let Res::Def(DefKind::Const | DefKind::AssocConst, item_def_id) = cx.qpath_res(qpath, expr.hir_id) else {
                return;
            };

            // Climb up to resolve any field access and explicit referencing.
            let mut cur_expr = expr;
            let mut dereferenced_expr = expr;
            let mut needs_check_adjustment = true;
            loop {
                let parent_id = cx.tcx.parent_hir_id(cur_expr.hir_id);
                if parent_id == cur_expr.hir_id {
                    break;
                }
                if let Node::Expr(parent_expr) = cx.tcx.hir_node(parent_id) {
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
                        ExprKind::Index(e, _, _) if ptr::eq(&raw const **e, cur_expr) => {
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

            if self.interior_mut.is_interior_mut_ty(cx, ty)
                && Self::is_value_unfrozen_expr(cx, expr.hir_id, item_def_id, ty)
            {
                lint(cx, Source::Expr { expr: expr.span });
            }
        }
    }
}

fn ignored_macro(cx: &LateContext<'_>, it: &Item<'_>) -> bool {
    macro_backtrace(it.span).any(|macro_call| {
        matches!(
            cx.tcx.get_diagnostic_name(macro_call.def_id),
            Some(sym::thread_local_macro)
        )
    })
}
