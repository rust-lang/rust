use clippy_utils::diagnostics::span_lint;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, FnRetTy, GenericArg, GenericBound, ItemKind, TraitBoundModifier, TyKind};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, ClauseKind, TyCtxt};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Looks for bounds in `impl Trait` in return position that are implied by other bounds.
    /// This is usually the case when a supertrait is explicitly specified, when it is already implied
    /// by a subtrait (e.g. `DerefMut: Deref`, so specifying `Deref` is unnecessary when `DerefMut` is specified).
    ///
    /// ### Why is this bad?
    /// Unnecessary complexity.
    ///
    /// ### Known problems
    /// This lint does not transitively look for implied bounds past the first supertrait.
    ///
    /// ### Example
    /// ```rust
    /// # use std::ops::{Deref,DerefMut};
    /// fn f() -> impl Deref<Target = i32> + DerefMut<Target = i32> {
    /// //             ^^^^^^^^^^^^^^^^^^^ unnecessary bound, already implied by the `DerefMut` trait bound
    ///     Box::new(123)
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::ops::{Deref,DerefMut};
    /// fn f() -> impl DerefMut<Target = i32> {
    ///     Box::new(123)
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub IMPLIED_BOUNDS_IN_IMPL,
    complexity,
    "specifying bounds that are implied by other bounds in `impl Trait` type"
}
declare_lint_pass!(ImpliedBoundsInImpl => [IMPLIED_BOUNDS_IN_IMPL]);

/// This function tries to, for all type parameters in a supertype predicate `GenericTrait<U>`,
/// check if the substituted type in the implied-by bound matches with what's subtituted in the
/// implied type.
///
/// Consider this example function.
/// ```rust,ignore
/// trait GenericTrait<T> {}
/// trait GenericSubTrait<T, U, V>: GenericTrait<U> {}
///                                              ^ trait_predicate_args: [Self#0, U#2]
/// impl GenericTrait<i32> for () {}
/// impl GenericSubTrait<(), i32, ()> for () {}
/// impl GenericSubTrait<(), [u8; 8], ()> for () {}
///
/// fn f() -> impl GenericTrait<i32> + GenericSubTrait<(), [u8; 8], ()> {
///                             ^^^ implied_args       ^^^^^^^^^^^^^^^ implied_by_args
///                                                                    (we are interested in `[u8; 8]` specifically, as that
///                                                                     is what `U` in `GenericTrait<U>` is substituted with)
///     ()
/// }
/// ```
/// Here i32 != [u8; 8], so this will return false.
fn is_same_generics(
    tcx: TyCtxt<'_>,
    trait_predicate_args: &[ty::GenericArg<'_>],
    implied_by_args: &[GenericArg<'_>],
    implied_args: &[GenericArg<'_>],
) -> bool {
    trait_predicate_args
        .iter()
        .enumerate()
        .skip(1) // skip `Self` implicit arg
        .all(|(arg_index, arg)| {
            if let Some(ty) = arg.as_type()
                && let &ty::Param(ty::ParamTy{ index, .. }) = ty.kind()
                // Since `trait_predicate_args` and type params in traits start with `Self=0`
                // and generic argument lists `GenericTrait<i32>` don't have `Self`,
                // we need to subtract 1 from the index.
                && let GenericArg::Type(ty_a) = implied_by_args[index as usize - 1]
                && let GenericArg::Type(ty_b) = implied_args[arg_index - 1]
            {
                hir_ty_to_ty(tcx, ty_a) == hir_ty_to_ty(tcx, ty_b)
            } else {
                false
            }
        })
}

impl LateLintPass<'_> for ImpliedBoundsInImpl {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        _: FnKind<'_>,
        decl: &FnDecl<'_>,
        _: &Body<'_>,
        _: Span,
        _: LocalDefId,
    ) {
        if let FnRetTy::Return(ty) = decl.output {
            if let TyKind::OpaqueDef(item_id, ..) = ty.kind
                && let item = cx.tcx.hir().item(item_id)
                && let ItemKind::OpaqueTy(opaque_ty) = item.kind
            {
                // Very often there is only a single bound, e.g. `impl Deref<..>`, in which case
                // we can avoid doing a bunch of stuff unnecessarily.
                if opaque_ty.bounds.is_empty() {
                    return;
                }

                // Get all the (implied) trait predicates in the bounds.
                // For `impl Deref + DerefMut` this will contain [`Deref`].
                // The implied `Deref` comes from `DerefMut` because `trait DerefMut: Deref {}`.

                // N.B. (G)ATs are fine to disregard, because they must be the same for all of its supertraits.
                // Example:
                // `impl Deref<Target = i32> + DerefMut<Target = u32>` is not allowed.
                // `DerefMut::Target` needs to match `Deref::Target`
                let implied_bounds: Vec<_> = opaque_ty.bounds.iter().filter_map(|bound| {
                    if let GenericBound::Trait(poly_trait, TraitBoundModifier::None) = bound
                        && let [.., path]  = poly_trait.trait_ref.path.segments
                        && poly_trait.bound_generic_params.is_empty()
                        && let Some(trait_def_id) = path.res.opt_def_id()
                        && let predicates = cx.tcx.super_predicates_of(trait_def_id).predicates
                        && !predicates.is_empty() // If the trait has no supertrait, there is nothing to add.
                    {
                        Some((path.args.map_or([].as_slice(), |a| a.args), predicates))
                    } else {
                        None
                    }
                }).collect();

                // Lint all bounds in the `impl Trait` type that are also in the `implied_bounds` vec.
                // This involves some extra logic when generic arguments are present, since
                // simply comparing trait `DefId`s won't be enough. We also need to compare the generics.
                for bound in opaque_ty.bounds {
                    if let GenericBound::Trait(poly_trait, TraitBoundModifier::None) = bound
                        && let [.., path] = poly_trait.trait_ref.path.segments
                        && let implied_args = path.args.map_or([].as_slice(), |a| a.args)
                        && let Some(def_id) = poly_trait.trait_ref.path.res.opt_def_id()
                        && implied_bounds.iter().any(|(implied_by_args, preds)| {
                            preds.iter().any(|(clause, _)| {
                                if let ClauseKind::Trait(tr) = clause.kind().skip_binder()
                                    && tr.def_id() == def_id
                                {
                                    is_same_generics(cx.tcx, tr.trait_ref.args, implied_by_args, implied_args)
                                } else {
                                    false
                                }
                            })
                        })
                    {
                        span_lint(cx, IMPLIED_BOUNDS_IN_IMPL, poly_trait.span, "this bound is implied by another bound and can be removed");
                    }
                }
            }
        }
    }
}
