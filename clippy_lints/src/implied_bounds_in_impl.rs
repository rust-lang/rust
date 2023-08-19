use clippy_utils::diagnostics::span_lint;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, FnRetTy, GenericArgs, GenericBound, ItemKind, TraitBoundModifier, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::ClauseKind;
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
    /// This lint currently does not work with generic traits (i.e. will not lint even if redundant).
    ///
    /// ### Example
    /// ```rust
    /// fn f() -> impl Deref<Target = i32> + DerefMut<Target = i32> {
    /// //             ^^^^^^^^^^^^^^^^^^^ unnecessary bound, already implied by the `DerefMut` trait bound
    ///     Box::new(123)
    /// }
    /// ```
    /// Use instead:
    /// ```rust
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
                // Get all `DefId`s of (implied) trait predicates in all the bounds.
                // For `impl Deref + DerefMut` this will contain [`Deref`].
                // The implied `Deref` comes from `DerefMut` because `trait DerefMut: Deref {}`.

                // N.B. Generic args on trait bounds are currently ignored and (G)ATs are fine to disregard,
                // because they must be the same for all of its supertraits. Example:
                // `impl Deref<Target = i32> + DerefMut<Target = u32>` is not allowed.
                // `DerefMut::Target` needs to match `Deref::Target`
                let implied_bounds = opaque_ty.bounds.iter().flat_map(|bound| {
                    if let GenericBound::Trait(poly_trait, TraitBoundModifier::None) = bound
                        && let [.., path]  = poly_trait.trait_ref.path.segments
                        && poly_trait.bound_generic_params.is_empty()
                        && path.args.map_or(true, GenericArgs::is_empty)
                        && let Some(trait_def_id) = path.res.opt_def_id()
                    {
                        cx.tcx.implied_predicates_of(trait_def_id).predicates
                    } else {
                        &[]
                    }
                }).collect::<Vec<_>>();

                // Lint all bounds in the `impl Trait` type that are also in the `implied_bounds` vec.
                for bound in opaque_ty.bounds {
                    if let GenericBound::Trait(poly_trait, TraitBoundModifier::None) = bound
                        && let Some(def_id) = poly_trait.trait_ref.path.res.opt_def_id()
                        && implied_bounds.iter().any(|(clause, _)| {
                            if let ClauseKind::Trait(tr) = clause.kind().skip_binder() {
                                tr.def_id() == def_id
                            } else {
                                false
                            }
                        })
                    {
                        span_lint(cx, IMPLIED_BOUNDS_IN_IMPL, poly_trait.span, "this bound is implied by another bound and can be removed");
                    }
                }
            }
        }
    }
}
