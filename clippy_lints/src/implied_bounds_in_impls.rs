use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_errors::{Applicability, SuggestionStyle};
use rustc_hir::def_id::DefId;
use rustc_hir::{
    AmbigArg, AssocItemConstraint, GenericArg, GenericBound, GenericBounds, PredicateOrigin, TraitBoundModifiers,
    TyKind, WherePredicateKind,
};
use rustc_hir_analysis::lower_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, AssocItem, ClauseKind, Generics, Ty, TyCtxt};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Looks for bounds in `impl Trait` in return position that are implied by other bounds.
    /// This can happen when a trait is specified that another trait already has as a supertrait
    /// (e.g. `fn() -> impl Deref + DerefMut<Target = i32>` has an unnecessary `Deref` bound,
    /// because `Deref` is a supertrait of `DerefMut`)
    ///
    /// ### Why is this bad?
    /// Specifying more bounds than necessary adds needless complexity for the reader.
    ///
    /// ### Limitations
    /// This lint does not check for implied bounds transitively. Meaning that
    /// it doesn't check for implied bounds from supertraits of supertraits
    /// (e.g. `trait A {} trait B: A {} trait C: B {}`, then having an `fn() -> impl A + C`)
    ///
    /// ### Example
    /// ```no_run
    /// # use std::ops::{Deref,DerefMut};
    /// fn f() -> impl Deref<Target = i32> + DerefMut<Target = i32> {
    /// //             ^^^^^^^^^^^^^^^^^^^ unnecessary bound, already implied by the `DerefMut` trait bound
    ///     Box::new(123)
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::ops::{Deref,DerefMut};
    /// fn f() -> impl DerefMut<Target = i32> {
    ///     Box::new(123)
    /// }
    /// ```
    #[clippy::version = "1.74.0"]
    pub IMPLIED_BOUNDS_IN_IMPLS,
    complexity,
    "specifying bounds that are implied by other bounds in `impl Trait` type"
}
declare_lint_pass!(ImpliedBoundsInImpls => [IMPLIED_BOUNDS_IN_IMPLS]);

fn emit_lint(
    cx: &LateContext<'_>,
    poly_trait: &rustc_hir::PolyTraitRef<'_>,
    bounds: GenericBounds<'_>,
    index: usize,
    // The constraints that were implied, used for suggestion purposes since removing a bound with
    // associated types means we might need to then move it to a different bound.
    implied_constraints: &[AssocItemConstraint<'_>],
    bound: &ImplTraitBound<'_>,
) {
    let implied_by = snippet(cx, bound.span, "..");

    span_lint_and_then(
        cx,
        IMPLIED_BOUNDS_IN_IMPLS,
        poly_trait.span,
        format!("this bound is already specified as the supertrait of `{implied_by}`"),
        |diag| {
            // If we suggest removing a bound, we may also need to extend the span
            // to include the `+` token that is ahead or behind,
            // so we don't end up with something like `impl + B` or `impl A + `

            let implied_span_extended = if let Some(next_bound) = bounds.get(index + 1) {
                poly_trait.span.to(next_bound.span().shrink_to_lo())
            } else if index > 0
                && let Some(prev_bound) = bounds.get(index - 1)
            {
                prev_bound.span().shrink_to_hi().to(poly_trait.span.shrink_to_hi())
            } else {
                poly_trait.span
            };

            let mut sugg = vec![(implied_span_extended, String::new())];

            // We also might need to include associated item constraints that were specified in the implied
            // bound, but omitted in the implied-by bound:
            // `fn f() -> impl Deref<Target = u8> + DerefMut`
            // If we're going to suggest removing `Deref<..>`, we'll need to put `<Target = u8>` on `DerefMut`
            let omitted_constraints: Vec<_> = implied_constraints
                .iter()
                .filter(|constraint| !bound.constraints.iter().any(|c| c.ident == constraint.ident))
                .collect();

            if !omitted_constraints.is_empty() {
                // `<>` needs to be added if there aren't yet any generic arguments or constraints
                let needs_angle_brackets = bound.args.is_empty() && bound.constraints.is_empty();
                let insert_span = match (bound.args, bound.constraints) {
                    ([.., arg], [.., constraint]) => arg.span().max(constraint.span).shrink_to_hi(),
                    ([.., arg], []) => arg.span().shrink_to_hi(),
                    ([], [.., constraint]) => constraint.span.shrink_to_hi(),
                    ([], []) => bound.span.shrink_to_hi(),
                };

                let mut constraints_sugg = if needs_angle_brackets {
                    "<".to_owned()
                } else {
                    // If angle brackets aren't needed (i.e., there are already generic arguments or constraints),
                    // we need to add a comma:
                    // `impl A<B, C >`
                    //             ^ if we insert `Assoc=i32` without a comma here, that'd be invalid syntax:
                    // `impl A<B, C Assoc=i32>`
                    ", ".to_owned()
                };

                for (index, constraint) in omitted_constraints.into_iter().enumerate() {
                    if index > 0 {
                        constraints_sugg += ", ";
                    }
                    constraints_sugg += &snippet(cx, constraint.span, "..");
                }
                if needs_angle_brackets {
                    constraints_sugg += ">";
                }
                sugg.push((insert_span, constraints_sugg));
            }

            diag.multipart_suggestion_with_style(
                "try removing this bound",
                sugg,
                Applicability::MachineApplicable,
                SuggestionStyle::ShowAlways,
            );
        },
    );
}

/// Tries to "resolve" a type.
/// The index passed to this function must start with `Self=0`, i.e. it must be a valid
/// type parameter index.
/// If the index is out of bounds, it means that the generic parameter has a default type.
fn try_resolve_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: &'tcx [GenericArg<'tcx>],
    generics: &'tcx Generics,
    index: usize,
) -> Option<Ty<'tcx>> {
    match args.get(index - 1) {
        // I don't think we care about `GenericArg::Infer` since this is all for stuff in type signatures
        // which do not permit inference variables.
        Some(GenericArg::Type(ty)) => Some(lower_ty(tcx, ty.as_unambig_ty())),
        Some(_) => None,
        None => Some(tcx.type_of(generics.own_params[index].def_id).skip_binder()),
    }
}

/// This function tries to, for all generic type parameters in a supertrait predicate `trait ...<U>:
/// GenericTrait<U>`, check if the substituted type in the implied-by bound matches with what's
/// substituted in the implied bound.
///
/// Consider this example.
/// ```rust,ignore
/// trait GenericTrait<T> {}
/// trait GenericSubTrait<T, U, V>: GenericTrait<U> {}
///                                 ^^^^^^^^^^^^^^^ trait_predicate_args: [Self#0, U#2]
///                                                 (the Self#0 is implicit: `<Self as GenericTrait<U>>`)
/// impl GenericTrait<i32> for () {}
/// impl GenericSubTrait<(), i32, ()> for () {}
/// impl GenericSubTrait<(), i64, ()> for () {}
///
/// fn f() -> impl GenericTrait<i32> + GenericSubTrait<(), i64, ()> {
///                             ^^^ implied_args       ^^^^^^^^^^^ implied_by_args
///                                                                (we are interested in `i64` specifically, as that
///                                                                 is what `U` in `GenericTrait<U>` is substituted with)
/// }
/// ```
/// Here i32 != i64, so this will return false.
fn is_same_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_predicate_args: &'tcx [ty::GenericArg<'tcx>],
    implied_by_args: &'tcx [GenericArg<'tcx>],
    implied_args: &'tcx [GenericArg<'tcx>],
    implied_by_def_id: DefId,
    implied_def_id: DefId,
) -> bool {
    // Get the generics of the two traits to be able to get default generic parameter.
    let implied_by_generics = tcx.generics_of(implied_by_def_id);
    let implied_generics = tcx.generics_of(implied_def_id);

    trait_predicate_args
        .iter()
        .enumerate()
        .skip(1) // skip `Self` implicit arg
        .all(|(arg_index, arg)| {
            if let Some(ty) = arg.as_type() {
                if let &ty::Param(ty::ParamTy { index, .. }) = ty.kind()
                    // `index == 0` means that it's referring to `Self`,
                    // in which case we don't try to substitute it
                    && index != 0
                    && let Some(ty_a) = try_resolve_type(tcx, implied_by_args, implied_by_generics, index as usize)
                    && let Some(ty_b) = try_resolve_type(tcx, implied_args, implied_generics, arg_index)
                {
                    ty_a == ty_b
                } else if let Some(ty_b) = try_resolve_type(tcx, implied_args, implied_generics, arg_index) {
                    ty == ty_b
                } else {
                    false
                }
            } else {
                false
            }
        })
}

struct ImplTraitBound<'tcx> {
    /// The span of the bound in the `impl Trait` type
    span: Span,
    /// The predicates defined in the trait referenced by this bound. This also contains the actual
    /// supertrait bounds
    predicates: &'tcx [(ty::Clause<'tcx>, Span)],
    /// The `DefId` of the trait being referenced by this bound
    trait_def_id: DefId,
    /// The generic arguments on the `impl Trait` bound
    args: &'tcx [GenericArg<'tcx>],
    /// The associated item constraints of this bound
    constraints: &'tcx [AssocItemConstraint<'tcx>],
}

/// Given an `impl Trait` type, gets all the supertraits from each bound ("implied bounds").
///
/// For `impl Deref + DerefMut + Eq` this returns `[Deref, PartialEq]`.
/// The `Deref` comes from `DerefMut` because `trait DerefMut: Deref {}`, and `PartialEq` comes from
/// `Eq`.
fn collect_supertrait_bounds<'tcx>(cx: &LateContext<'tcx>, bounds: GenericBounds<'tcx>) -> Vec<ImplTraitBound<'tcx>> {
    bounds
        .iter()
        .filter_map(|bound| {
            if let GenericBound::Trait(poly_trait) = bound
                && let TraitBoundModifiers::NONE = poly_trait.modifiers
                && let [.., path] = poly_trait.trait_ref.path.segments
                && poly_trait.bound_generic_params.is_empty()
                && let Some(trait_def_id) = path.res.opt_def_id()
                && let predicates = cx.tcx.explicit_super_predicates_of(trait_def_id).skip_binder()
                // If the trait has no supertrait, there is no need to collect anything from that bound
                && !predicates.is_empty()
            {
                Some(ImplTraitBound {
                    span: bound.span(),
                    predicates,
                    trait_def_id,
                    args: path.args.map_or([].as_slice(), |p| p.args),
                    constraints: path.args.map_or([].as_slice(), |p| p.constraints),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Given a bound in an `impl Trait` type, looks for a trait in the set of supertraits (previously
/// collected in [`collect_supertrait_bounds`]) that matches (same trait and generic arguments).
fn find_bound_in_supertraits<'a, 'tcx>(
    cx: &LateContext<'tcx>,
    trait_def_id: DefId,
    args: &'tcx [GenericArg<'tcx>],
    bounds: &'a [ImplTraitBound<'tcx>],
) -> Option<&'a ImplTraitBound<'tcx>> {
    bounds.iter().find(|bound| {
        bound.predicates.iter().any(|(clause, _)| {
            if let ClauseKind::Trait(tr) = clause.kind().skip_binder()
                && tr.def_id() == trait_def_id
            {
                is_same_generics(
                    cx.tcx,
                    tr.trait_ref.args,
                    bound.args,
                    args,
                    bound.trait_def_id,
                    trait_def_id,
                )
            } else {
                false
            }
        })
    })
}

fn check<'tcx>(cx: &LateContext<'tcx>, bounds: GenericBounds<'tcx>) {
    if bounds.len() == 1 {
        // Very often there is only a single bound, e.g. `impl Deref<..>`, in which case
        // we can avoid doing a bunch of stuff unnecessarily; there will trivially be
        // no duplicate bounds
        return;
    }

    let supertraits = collect_supertrait_bounds(cx, bounds);

    // Lint all bounds in the `impl Trait` type that we've previously also seen in the set of
    // supertraits of each of the bounds.
    // This involves some extra logic when generic arguments are present, since
    // simply comparing trait `DefId`s won't be enough. We also need to compare the generics.
    for (index, bound) in bounds.iter().enumerate() {
        if let GenericBound::Trait(poly_trait) = bound
            && let TraitBoundModifiers::NONE = poly_trait.modifiers
            && let [.., path] = poly_trait.trait_ref.path.segments
            && let implied_args = path.args.map_or([].as_slice(), |a| a.args)
            && let implied_constraints = path.args.map_or([].as_slice(), |a| a.constraints)
            && let Some(def_id) = poly_trait.trait_ref.path.res.opt_def_id()
            && let Some(bound) = find_bound_in_supertraits(cx, def_id, implied_args, &supertraits)
            // If the implied bound has a type binding that also exists in the implied-by trait,
            // then we shouldn't lint. See #11880 for an example.
            && let assocs = cx.tcx.associated_items(bound.trait_def_id)
            && !implied_constraints.iter().any(|constraint| {
                assocs
                    .filter_by_name_unhygienic(constraint.ident.name)
                    .next()
                    .is_some_and(AssocItem::is_type)
                })
        {
            emit_lint(cx, poly_trait, bounds, index, implied_constraints, bound);
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ImpliedBoundsInImpls {
    fn check_generics(&mut self, cx: &LateContext<'tcx>, generics: &rustc_hir::Generics<'tcx>) {
        for predicate in generics.predicates {
            if let WherePredicateKind::BoundPredicate(predicate) = predicate.kind
                // In theory, the origin doesn't really matter,
                // we *could* also lint on explicit where clauses written out by the user,
                // not just impl trait desugared ones, but that contradicts with the lint name...
                && let PredicateOrigin::ImplTrait = predicate.origin
            {
                check(cx, predicate.bounds);
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &rustc_hir::Ty<'tcx, AmbigArg>) {
        if let TyKind::OpaqueDef(opaque_ty, ..) = ty.kind {
            check(cx, opaque_ty.bounds);
        }
    }
}
