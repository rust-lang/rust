//! Code that handles "type-outlives" constraints like `T: 'a`. This
//! is based on the `compute_outlives_components` function defined in rustc_infer,
//! but it adds a bit of heuristics on top, in particular to deal with
//! associated types and projections.
//!
//! When we process a given `T: 'a` obligation, we may produce two
//! kinds of constraints for the region inferencer:
//!
//! - Relationships between inference variables and other regions.
//!   For example, if we have `&'?0 u32: 'a`, then we would produce
//!   a constraint that `'a <= '?0`.
//! - "Verifys" that must be checked after inferencing is done.
//!   For example, if we know that, for some type parameter `T`,
//!   `T: 'a + 'b`, and we have a requirement that `T: '?1`,
//!   then we add a "verify" that checks that `'?1 <= 'a || '?1 <= 'b`.
//!   - Note the difference with the previous case: here, the region
//!     variable must be less than something else, so this doesn't
//!     affect how inference works (it finds the smallest region that
//!     will do); it's just a post-condition that we have to check.
//!
//! **The key point is that once this function is done, we have
//! reduced all of our "type-region outlives" obligations into relationships
//! between individual regions.**
//!
//! One key input to this function is the set of "region-bound pairs".
//! These are basically the relationships between type parameters and
//! regions that are in scope at the point where the outlives
//! obligation was incurred. **When type-checking a function,
//! particularly in the face of closures, this is not known until
//! regionck runs!** This is because some of those bounds come
//! from things we have yet to infer.
//!
//! Consider:
//!
//! ```
//! fn bar<T>(a: T, b: impl for<'a> Fn(&'a T)) {}
//! fn foo<T>(x: T) {
//!     bar(x, |y| { /* ... */})
//!     //      ^ closure arg
//! }
//! ```
//!
//! Here, the type of `y` may involve inference variables and the
//! like, and it may also contain implied bounds that are needed to
//! type-check the closure body (e.g., here it informs us that `T`
//! outlives the late-bound region `'a`).
//!
//! Note that by delaying the gathering of implied bounds until all
//! inference information is known, we may find relationships between
//! bound regions and other regions in the environment. For example,
//! when we first check a closure like the one expected as argument
//! to `foo`:
//!
//! ```
//! fn foo<U, F: for<'a> FnMut(&'a U)>(_f: F) {}
//! ```
//!
//! the type of the closure's first argument would be `&'a ?U`. We
//! might later infer `?U` to something like `&'b u32`, which would
//! imply that `'b: 'a`.

use rustc_data_structures::assert_matches;
use rustc_middle::bug;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::outlives::{
    Component, compute_alias_components_recursive, compute_outlives_components,
};
use rustc_middle::ty::{
    self, GenericArgKind, GenericArgsRef, OutlivesPredicate, PolyTypeOutlivesPredicate, Ty, TyCtxt,
    TypeFoldable as _, TypeVisitableExt,
};
use tracing::{debug, instrument, trace};

use super::env::OutlivesEnvironment;
use crate::infer::outlives::env::RegionBoundPairs;
use crate::infer::region_constraints::VerifyIfEq;
use crate::infer::resolve::OpportunisticRegionResolver;
use crate::infer::{
    self, GenericKind, InferCtxt, SubregionOrigin, TypeOutlivesConstraint, VerifyBound,
};

impl<'tcx> InferCtxt<'tcx> {
    /// Process the region obligations that must be proven (during
    /// `regionck`) for the given `body_id`, given information about
    /// the region bounds in scope and so forth.
    ///
    /// See the `region_obligations` field of `InferCtxt` for some
    /// comments about how this function fits into the overall expected
    /// flow of the inferencer. The key point is that it is
    /// invoked after all type-inference variables have been bound --
    /// right before lexical region resolution.
    #[instrument(level = "debug", skip(self, outlives_env, deeply_normalize_ty))]
    pub fn process_registered_region_obligations(
        &self,
        outlives_env: &OutlivesEnvironment<'tcx>,
        mut deeply_normalize_ty: impl FnMut(
            PolyTypeOutlivesPredicate<'tcx>,
            SubregionOrigin<'tcx>,
        )
            -> Result<PolyTypeOutlivesPredicate<'tcx>, NoSolution>,
    ) -> Result<(), (PolyTypeOutlivesPredicate<'tcx>, SubregionOrigin<'tcx>)> {
        assert!(!self.in_snapshot(), "cannot process registered region obligations in a snapshot");

        // Must loop since the process of normalizing may itself register region obligations.
        for iteration in 0.. {
            let my_region_obligations = self.take_registered_region_obligations();
            if my_region_obligations.is_empty() {
                break;
            }

            if !self.tcx.recursion_limit().value_within_limit(iteration) {
                // This may actually be reachable. If so, we should convert
                // this to a proper error/consider whether we should detect
                // this somewhere else.
                bug!(
                    "unexpected overflowed when processing region obligations: {my_region_obligations:#?}"
                );
            }

            for TypeOutlivesConstraint { sup_type, sub_region, origin } in my_region_obligations {
                let outlives = ty::Binder::dummy(ty::OutlivesPredicate(sup_type, sub_region));
                let ty::OutlivesPredicate(sup_type, sub_region) =
                    deeply_normalize_ty(outlives, origin.clone())
                        .map_err(|NoSolution| (outlives, origin.clone()))?
                        .no_bound_vars()
                        .expect("started with no bound vars, should end with no bound vars");
                // `TypeOutlives` is structural, so we should try to opportunistically resolve all
                // region vids before processing regions, so we have a better chance to match clauses
                // in our param-env.
                let (sup_type, sub_region) =
                    (sup_type, sub_region).fold_with(&mut OpportunisticRegionResolver::new(self));

                if self.tcx.sess.opts.unstable_opts.higher_ranked_assumptions
                    && outlives_env
                        .higher_ranked_assumptions()
                        .contains(&ty::OutlivesPredicate(sup_type.into(), sub_region))
                {
                    continue;
                }

                debug!(?sup_type, ?sub_region, ?origin);

                let category = origin.to_constraint_category();
                require_type_outlives(
                    &mut TypeOutlivesOpCtxt::new(
                        self,
                        self.tcx,
                        outlives_env.region_bound_pairs(),
                        None,
                        outlives_env.known_type_outlives(),
                    ),
                    origin,
                    sup_type,
                    sub_region,
                    category,
                );
            }
        }

        Ok(())
    }
}

/// Context used for constructing a [`VerifyBound`]. See module comments
/// for more information
pub(crate) struct VerifyBoundCx<'cx, 'tcx> {
    tcx: TyCtxt<'tcx>,
    region_bound_pairs: &'cx RegionBoundPairs<'tcx>,
    /// During borrowck, if there are no outlives bounds on a generic
    /// parameter `T`, we assume that `T: 'in_fn_body` holds.
    ///
    /// Outside of borrowck the only way to prove `T: '?0` is by
    /// setting  `'?0` to `'empty`.
    implicit_region_bound: Option<ty::Region<'tcx>>,
    caller_bounds: &'cx [ty::PolyTypeOutlivesPredicate<'tcx>],
}

impl<'cx, 'tcx> VerifyBoundCx<'cx, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        region_bound_pairs: &'cx RegionBoundPairs<'tcx>,
        implicit_region_bound: Option<ty::Region<'tcx>>,
        caller_bounds: &'cx [ty::PolyTypeOutlivesPredicate<'tcx>],
    ) -> Self {
        Self { tcx, region_bound_pairs, implicit_region_bound, caller_bounds }
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn param_or_placeholder_bound(&self, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        // Start with anything like `T: 'a` we can scrape from the
        // environment. If the environment contains something like
        // `for<'a> T: 'a`, then we know that `T` outlives everything.
        let declared_bounds_from_env = self.declared_generic_bounds_from_env(ty);
        debug!(?declared_bounds_from_env);
        let mut param_bounds = vec![];
        for declared_bound in declared_bounds_from_env {
            let bound_region = declared_bound.map_bound(|outlives| outlives.1);
            if let Some(region) = bound_region.no_bound_vars() {
                // This is `T: 'a` for some free region `'a`.
                param_bounds.push(VerifyBound::OutlivedBy(region));
            } else {
                // This is `for<'a> T: 'a`. This means that `T` outlives everything! All done here.
                debug!("found that {ty:?} outlives any lifetime, returning empty vector");
                return VerifyBound::AllBounds(vec![]);
            }
        }

        // Add in the default bound of fn body that applies to all in
        // scope type parameters:
        if let Some(r) = self.implicit_region_bound {
            debug!("adding implicit region bound of {r:?}");
            param_bounds.push(VerifyBound::OutlivedBy(r));
        }

        if param_bounds.is_empty() {
            // We know that all types `T` outlive `'empty`, so if we
            // can find no other bound, then check that the region
            // being tested is `'empty`.
            VerifyBound::IsEmpty
        } else if param_bounds.len() == 1 {
            // Micro-opt: no need to store the vector if it's just len 1
            param_bounds.pop().unwrap()
        } else {
            // If we can find any other bound `R` such that `T: R`, then
            // we don't need to check for `'empty`, because `R: 'empty`.
            VerifyBound::AnyBound(param_bounds)
        }
    }

    /// Given a projection like `T::Item`, searches the environment
    /// for where-clauses like `T::Item: 'a`. Returns the set of
    /// regions `'a` that it finds.
    ///
    /// This is an "approximate" check -- it may not find all
    /// applicable bounds, and not all the bounds it returns can be
    /// relied upon. In particular, this check ignores region
    /// identity. So, for example, if we have `<T as
    /// Trait<'0>>::Item` where `'0` is a region variable, and the
    /// user has `<T as Trait<'a>>::Item: 'b` in the environment, then
    /// the clause from the environment only applies if `'0 = 'a`,
    /// which we don't know yet. But we would still include `'b` in
    /// this list.
    pub(crate) fn approx_declared_bounds_from_env(
        &self,
        alias_ty: ty::AliasTy<'tcx>,
    ) -> Vec<ty::PolyTypeOutlivesPredicate<'tcx>> {
        let erased_alias_ty = self.tcx.erase_and_anonymize_regions(alias_ty.to_ty(self.tcx));
        self.declared_generic_bounds_from_env_for_erased_ty(erased_alias_ty)
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn alias_ty_bound(&self, alias_ty: ty::AliasTy<'tcx>) -> VerifyBound<'tcx> {
        // Search the env for where clauses like `P: 'a`.
        let env_bounds = self.approx_declared_bounds_from_env(alias_ty).into_iter().map(|binder| {
            if let Some(ty::OutlivesPredicate(ty, r)) = binder.no_bound_vars()
                && let ty::Alias(_, alias_ty_from_bound) = *ty.kind()
                && alias_ty_from_bound == alias_ty
            {
                // Micro-optimize if this is an exact match (this
                // occurs often when there are no region variables
                // involved).
                VerifyBound::OutlivedBy(r)
            } else {
                let verify_if_eq_b =
                    binder.map_bound(|ty::OutlivesPredicate(ty, bound)| VerifyIfEq { ty, bound });
                VerifyBound::IfEq(verify_if_eq_b)
            }
        });

        // Extend with bounds that we can find from the definition.
        let definition_bounds =
            self.declared_bounds_from_definition(alias_ty).map(|r| VerifyBound::OutlivedBy(r));

        // see the extensive comment in alias_ty_must_outlive
        let recursive_bound = {
            let kind = alias_ty.kind(self.tcx);
            let components = compute_alias_components_recursive(self.tcx, kind, alias_ty);
            self.bound_from_components(&components)
        };

        VerifyBound::AnyBound(env_bounds.chain(definition_bounds).collect()).or(recursive_bound)
    }

    fn bound_from_components(&self, components: &[Component<TyCtxt<'tcx>>]) -> VerifyBound<'tcx> {
        let mut bounds = components
            .iter()
            .map(|component| self.bound_from_single_component(component))
            // Remove bounds that must hold, since they are not interesting.
            .filter(|bound| !bound.must_hold());

        match (bounds.next(), bounds.next()) {
            (Some(first), None) => first,
            (first, second) => {
                VerifyBound::AllBounds(first.into_iter().chain(second).chain(bounds).collect())
            }
        }
    }

    fn bound_from_single_component(
        &self,
        component: &Component<TyCtxt<'tcx>>,
    ) -> VerifyBound<'tcx> {
        match *component {
            Component::Region(lt) => VerifyBound::OutlivedBy(lt),
            Component::Param(param_ty) => self.param_or_placeholder_bound(param_ty.to_ty(self.tcx)),
            Component::Placeholder(placeholder_ty) => {
                self.param_or_placeholder_bound(Ty::new_placeholder(self.tcx, placeholder_ty))
            }
            Component::Alias(alias_ty) => self.alias_ty_bound(alias_ty),
            Component::EscapingAlias(ref components) => self.bound_from_components(components),
            Component::UnresolvedInferenceVariable(v) => {
                // Ignore this, we presume it will yield an error later, since
                // if a type variable is not resolved by this point it never
                // will be.
                self.tcx
                    .dcx()
                    .delayed_bug(format!("unresolved inference variable in outlives: {v:?}"));
                // Add a bound that never holds.
                VerifyBound::AnyBound(vec![])
            }
        }
    }

    /// Searches the environment for where-clauses like `G: 'a` where
    /// `G` is either some type parameter `T` or a projection like
    /// `T::Item`. Returns a vector of the `'a` bounds it can find.
    ///
    /// This is a conservative check -- it may not find all applicable
    /// bounds, but all the bounds it returns can be relied upon.
    fn declared_generic_bounds_from_env(
        &self,
        generic_ty: Ty<'tcx>,
    ) -> Vec<ty::PolyTypeOutlivesPredicate<'tcx>> {
        assert_matches!(generic_ty.kind(), ty::Param(_) | ty::Placeholder(_));
        self.declared_generic_bounds_from_env_for_erased_ty(generic_ty)
    }

    /// Searches the environment to find all bounds that apply to `erased_ty`.
    /// Obviously these must be approximate -- they are in fact both *over* and
    /// and *under* approximated:
    ///
    /// * Over-approximated because we don't consider equality of regions.
    /// * Under-approximated because we look for syntactic equality and so for complex types
    ///   like `<T as Foo<fn(&u32, &u32)>>::Item` or whatever we may fail to figure out
    ///   all the subtleties.
    ///
    /// In some cases, such as when `erased_ty` represents a `ty::Param`, however,
    /// the result is precise.
    #[instrument(level = "debug", skip(self))]
    fn declared_generic_bounds_from_env_for_erased_ty(
        &self,
        erased_ty: Ty<'tcx>,
    ) -> Vec<ty::PolyTypeOutlivesPredicate<'tcx>> {
        let tcx = self.tcx;
        let mut bounds = vec![];

        // To start, collect bounds from user environment. Note that
        // parameter environments are already elaborated, so we don't
        // have to worry about that.
        bounds.extend(self.caller_bounds.iter().copied().filter(move |outlives_predicate| {
            super::test_type_match::can_match_erased_ty(tcx, *outlives_predicate, erased_ty)
        }));

        // Next, collect regions we scraped from the well-formedness
        // constraints in the fn signature. To do that, we walk the list
        // of known relations from the fn ctxt.
        //
        // This is crucial because otherwise code like this fails:
        //
        //     fn foo<'a, A>(x: &'a A) { x.bar() }
        //
        // The problem is that the type of `x` is `&'a A`. To be
        // well-formed, then, A must outlive `'a`, but we don't know that
        // this holds from first principles.
        bounds.extend(self.region_bound_pairs.iter().filter_map(|&OutlivesPredicate(p, r)| {
            debug!(
                "declared_generic_bounds_from_env_for_erased_ty: region_bound_pair = {:?}",
                (r, p)
            );
            // Fast path for the common case.
            match (&p, erased_ty.kind()) {
                // In outlive routines, all types are expected to be fully normalized.
                // And therefore we can safely use structural equality for alias types.
                (GenericKind::Param(p1), ty::Param(p2)) if p1 == p2 => {}
                (GenericKind::Placeholder(p1), ty::Placeholder(p2)) if p1 == p2 => {}
                (GenericKind::Alias(a1), ty::Alias(_, a2)) if a1.def_id == a2.def_id => {}
                _ => return None,
            }

            let p_ty = p.to_ty(tcx);
            let erased_p_ty = self.tcx.erase_and_anonymize_regions(p_ty);
            (erased_p_ty == erased_ty).then_some(ty::Binder::dummy(ty::OutlivesPredicate(p_ty, r)))
        }));

        bounds
    }

    /// Given a projection like `<T as Foo<'x>>::Bar`, returns any bounds
    /// declared in the trait definition. For example, if the trait were
    ///
    /// ```rust
    /// trait Foo<'a> {
    ///     type Bar: 'a;
    /// }
    /// ```
    ///
    /// If we were given the `DefId` of `Foo::Bar`, we would return
    /// `'a`. You could then apply the instantiations from the
    /// projection to convert this into your namespace. This also
    /// works if the user writes `where <Self as Foo<'a>>::Bar: 'a` on
    /// the trait. In fact, it works by searching for just such a
    /// where-clause.
    ///
    /// It will not, however, work for higher-ranked bounds like:
    ///
    /// ```ignore(this does compile today, previously was marked as `compile_fail,E0311`)
    /// trait Foo<'a, 'b>
    /// where for<'x> <Self as Foo<'x, 'b>>::Bar: 'x
    /// {
    ///     type Bar;
    /// }
    /// ```
    ///
    /// This is for simplicity, and because we are not really smart
    /// enough to cope with such bounds anywhere.
    pub(crate) fn declared_bounds_from_definition(
        &self,
        alias_ty: ty::AliasTy<'tcx>,
    ) -> impl Iterator<Item = ty::Region<'tcx>> {
        let tcx = self.tcx;
        let bounds = tcx.item_self_bounds(alias_ty.def_id);
        trace!("{:#?}", bounds.skip_binder());
        bounds
            .iter_instantiated(tcx, alias_ty.args)
            .filter_map(|p| p.as_type_outlives_clause())
            .filter_map(|p| p.no_bound_vars())
            .map(|OutlivesPredicate(_, r)| r)
    }
}

/// Holds necessary context for the [`require_type_outlives`] operation.
pub struct TypeOutlivesOpCtxt<'cx, 'tcx, D>
where
    D: OutlivesHandlingDelegate<'tcx>,
{
    delegate: D,
    tcx: TyCtxt<'tcx>,
    verify_bound_cx: VerifyBoundCx<'cx, 'tcx>,
}

pub trait OutlivesHandlingDelegate<'tcx> {
    /// Subtle: this trait exists to abstract the outlives handling between
    /// regular regionck and NLL. Unfortunately, NLL and regionck don't agree
    /// on how subtyping works.
    ///
    /// In NLL `'a sub 'b` means `'a outlives 'b`.
    /// In regionck `'a sub 'b` means the set of locations `'a` is live at is a subset
    /// of the locations that `'b` is, or in other words, `'b outlives 'a`.
    ///
    /// This method will be called with the regionck meaning of subtyping. i.e. if
    /// there is some `&'b u32: 'static` constraint, we will give a `'b sub 'static`
    /// constraint.
    fn push_sub_region_constraint(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
        constraint_category: ConstraintCategory<'tcx>,
    );

    fn push_verify(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        a: ty::Region<'tcx>,
        bound: VerifyBound<'tcx>,
    );
}

impl<'cx, 'tcx, D> TypeOutlivesOpCtxt<'cx, 'tcx, D>
where
    D: OutlivesHandlingDelegate<'tcx>,
{
    pub fn new(
        delegate: D,
        tcx: TyCtxt<'tcx>,
        region_bound_pairs: &'cx RegionBoundPairs<'tcx>,
        implicit_region_bound: Option<ty::Region<'tcx>>,
        caller_bounds: &'cx [ty::PolyTypeOutlivesPredicate<'tcx>],
    ) -> Self {
        Self {
            delegate,
            tcx,
            verify_bound_cx: VerifyBoundCx::new(
                tcx,
                region_bound_pairs,
                implicit_region_bound,
                caller_bounds,
            ),
        }
    }
}

/// "lowers" a `T: 'a` obligation into a series of `'a: 'b` constraints
/// and "verify"s, as described on the module comment. The final constraints
/// are emitted via a "delegate" of type `D` -- this is either the `infcx`, which
/// accrues them into the `region_obligations` code, or a `ConstraintConversion`
/// (used during borrow checking).
///
/// # Parameters
///
/// - `origin`, the reason we need this constraint
/// - `ty`, the type `T`
/// - `region`, the region `'a`
#[instrument(level = "debug", skip(ctxt))]
pub fn require_type_outlives<'tcx, D: OutlivesHandlingDelegate<'tcx>>(
    ctxt: &mut TypeOutlivesOpCtxt<'_, 'tcx, D>,
    // ideally this would be in `ctxt` but then we lose ownership
    origin: infer::SubregionOrigin<'tcx>,
    ty: Ty<'tcx>,
    region: ty::Region<'tcx>,
    category: ConstraintCategory<'tcx>,
) {
    assert!(!ty.has_escaping_bound_vars());

    let components = compute_outlives_components(ctxt.tcx, ty);
    ctxt.components_must_outlive(origin, &components, region, category);
}

impl<'tcx, D: OutlivesHandlingDelegate<'tcx>> TypeOutlivesOpCtxt<'_, 'tcx, D> {
    fn components_must_outlive(
        &mut self,
        origin: infer::SubregionOrigin<'tcx>,
        components: &[Component<TyCtxt<'tcx>>],
        region: ty::Region<'tcx>,
        category: ConstraintCategory<'tcx>,
    ) {
        for component in components.iter() {
            let origin = origin.clone();
            match component {
                Component::Region(region1) => {
                    self.delegate.push_sub_region_constraint(origin, region, *region1, category);
                }
                Component::Param(param_ty) => {
                    self.param_ty_must_outlive(origin, region, *param_ty);
                }
                Component::Placeholder(placeholder_ty) => {
                    self.placeholder_ty_must_outlive(origin, region, *placeholder_ty);
                }
                Component::Alias(alias_ty) => self.alias_ty_must_outlive(origin, region, *alias_ty),
                Component::EscapingAlias(subcomponents) => {
                    self.components_must_outlive(origin, subcomponents, region, category);
                }
                Component::UnresolvedInferenceVariable(v) => {
                    // Ignore this, we presume it will yield an error later,
                    // since if a type variable is not resolved by this point
                    // it never will be.
                    self.tcx.dcx().span_delayed_bug(
                        origin.span(),
                        format!("unresolved inference variable in outlives: {v:?}"),
                    );
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn param_ty_must_outlive(
        &mut self,
        origin: infer::SubregionOrigin<'tcx>,
        region: ty::Region<'tcx>,
        param_ty: ty::ParamTy,
    ) {
        let verify_bound =
            self.verify_bound_cx.param_or_placeholder_bound(param_ty.to_ty(self.tcx));
        self.delegate.push_verify(origin, GenericKind::Param(param_ty), region, verify_bound);
    }

    #[instrument(level = "debug", skip(self))]
    fn placeholder_ty_must_outlive(
        &mut self,
        origin: infer::SubregionOrigin<'tcx>,
        region: ty::Region<'tcx>,
        placeholder_ty: ty::PlaceholderType<'tcx>,
    ) {
        let verify_bound = self
            .verify_bound_cx
            .param_or_placeholder_bound(Ty::new_placeholder(self.tcx, placeholder_ty));
        self.delegate.push_verify(
            origin,
            GenericKind::Placeholder(placeholder_ty),
            region,
            verify_bound,
        );
    }

    #[instrument(level = "debug", skip(self))]
    fn alias_ty_must_outlive(
        &mut self,
        origin: infer::SubregionOrigin<'tcx>,
        region: ty::Region<'tcx>,
        alias_ty: ty::AliasTy<'tcx>,
    ) {
        // An optimization for a common case with opaque types.
        if alias_ty.args.is_empty() {
            return;
        }

        if alias_ty.has_non_region_infer() {
            self.tcx
                .dcx()
                .span_delayed_bug(origin.span(), "an alias has infers during region solving");
            return;
        }

        // This case is thorny for inference. The fundamental problem is
        // that there are many cases where we have choice, and inference
        // doesn't like choice (the current region inference in
        // particular). :) First off, we have to choose between using the
        // OutlivesProjectionEnv, OutlivesProjectionTraitDef, and
        // OutlivesProjectionComponent rules, any one of which is
        // sufficient. If there are no inference variables involved, it's
        // not hard to pick the right rule, but if there are, we're in a
        // bit of a catch 22: if we picked which rule we were going to
        // use, we could add constraints to the region inference graph
        // that make it apply, but if we don't add those constraints, the
        // rule might not apply (but another rule might). For now, we err
        // on the side of adding too few edges into the graph.

        // Compute the bounds we can derive from the trait definition.
        // These are guaranteed to apply, no matter the inference
        // results.
        let trait_bounds: Vec<_> =
            self.verify_bound_cx.declared_bounds_from_definition(alias_ty).collect();

        debug!(?trait_bounds);

        // Compute the bounds we can derive from the environment. This
        // is an "approximate" match -- in some cases, these bounds
        // may not apply.
        let approx_env_bounds = self.verify_bound_cx.approx_declared_bounds_from_env(alias_ty);
        debug!(?approx_env_bounds);

        // If declared bounds list is empty, the only applicable rule is
        // OutlivesProjectionComponent. If there are inference variables,
        // then, we can break down the outlives into more primitive
        // components without adding unnecessary edges.
        //
        // If there are *no* inference variables, however, we COULD do
        // this, but we choose not to, because the error messages are less
        // good. For example, a requirement like `T::Item: 'r` would be
        // translated to a requirement that `T: 'r`; when this is reported
        // to the user, it will thus say "T: 'r must hold so that T::Item:
        // 'r holds". But that makes it sound like the only way to fix
        // the problem is to add `T: 'r`, which isn't true. So, if there are no
        // inference variables, we use a verify constraint instead of adding
        // edges, which winds up enforcing the same condition.
        let kind = alias_ty.kind(self.tcx);
        if approx_env_bounds.is_empty()
            && trait_bounds.is_empty()
            && (alias_ty.has_infer_regions() || kind == ty::Opaque)
        {
            debug!("no declared bounds");
            let opt_variances = self.tcx.opt_alias_variances(kind, alias_ty.def_id);
            self.args_must_outlive(alias_ty.args, origin, region, opt_variances);
            return;
        }

        // If we found a unique bound `'b` from the trait, and we
        // found nothing else from the environment, then the best
        // action is to require that `'b: 'r`, so do that.
        //
        // This is best no matter what rule we use:
        //
        // - OutlivesProjectionEnv: these would translate to the requirement that `'b:'r`
        // - OutlivesProjectionTraitDef: these would translate to the requirement that `'b:'r`
        // - OutlivesProjectionComponent: this would require `'b:'r`
        //   in addition to other conditions
        if !trait_bounds.is_empty()
            && trait_bounds[1..]
                .iter()
                .map(|r| Some(*r))
                .chain(
                    // NB: The environment may contain `for<'a> T: 'a` style bounds.
                    // In that case, we don't know if they are equal to the trait bound
                    // or not (since we don't *know* whether the environment bound even applies),
                    // so just map to `None` here if there are bound vars, ensuring that
                    // the call to `all` will fail below.
                    approx_env_bounds.iter().map(|b| b.map_bound(|b| b.1).no_bound_vars()),
                )
                .all(|b| b == Some(trait_bounds[0]))
        {
            let unique_bound = trait_bounds[0];
            debug!(?unique_bound);
            debug!("unique declared bound appears in trait ref");
            let category = origin.to_constraint_category();
            self.delegate.push_sub_region_constraint(origin, region, unique_bound, category);
            return;
        }

        // Fallback to verifying after the fact that there exists a
        // declared bound, or that all the components appearing in the
        // projection outlive; in some cases, this may add insufficient
        // edges into the inference graph, leading to inference failures
        // even though a satisfactory solution exists.
        let verify_bound = self.verify_bound_cx.alias_ty_bound(alias_ty);
        debug!("alias_must_outlive: pushing {:?}", verify_bound);
        self.delegate.push_verify(origin, GenericKind::Alias(alias_ty), region, verify_bound);
    }

    #[instrument(level = "debug", skip(self))]
    fn args_must_outlive(
        &mut self,
        args: GenericArgsRef<'tcx>,
        origin: infer::SubregionOrigin<'tcx>,
        region: ty::Region<'tcx>,
        opt_variances: Option<&[ty::Variance]>,
    ) {
        let constraint = origin.to_constraint_category();
        for (index, arg) in args.iter().enumerate() {
            match arg.kind() {
                GenericArgKind::Lifetime(lt) => {
                    let variance = if let Some(variances) = opt_variances {
                        variances[index]
                    } else {
                        ty::Invariant
                    };
                    if variance == ty::Invariant {
                        self.delegate.push_sub_region_constraint(
                            origin.clone(),
                            region,
                            lt,
                            constraint,
                        );
                    }
                }
                GenericArgKind::Type(ty) => {
                    require_type_outlives(self, origin.clone(), ty, region, constraint);
                }
                GenericArgKind::Const(_) => {
                    // Const parameters don't impose constraints.
                }
            }
        }
    }
}

impl<'cx, 'tcx> OutlivesHandlingDelegate<'tcx> for &'cx InferCtxt<'tcx> {
    fn push_sub_region_constraint(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
        _constraint_category: ConstraintCategory<'tcx>,
    ) {
        self.sub_regions(origin, a, b)
    }

    fn push_verify(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        a: ty::Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) {
        self.verify_generic_bound(origin, kind, a, bound)
    }
}
