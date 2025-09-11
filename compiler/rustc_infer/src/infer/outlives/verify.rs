use std::assert_matches::assert_matches;

use rustc_middle::ty::outlives::{Component, compute_alias_components_recursive};
use rustc_middle::ty::{self, OutlivesPredicate, Ty, TyCtxt};
use smallvec::smallvec;
use tracing::{debug, instrument, trace};

use crate::infer::outlives::env::RegionBoundPairs;
use crate::infer::region_constraints::VerifyIfEq;
use crate::infer::{GenericKind, VerifyBound};

/// The `TypeOutlives` struct has the job of "lowering" a `T: 'a`
/// obligation into a series of `'a: 'b` constraints and "verifys", as
/// described on the module comment. The final constraints are emitted
/// via a "delegate" of type `D` -- this is usually the `infcx`, which
/// accrues them into the `region_obligations` code, but for NLL we
/// use something else.
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
    pub(crate) fn alias_bound(&self, alias_ty: ty::AliasTy<'tcx>) -> VerifyBound<'tcx> {
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

        // see the extensive comment in projection_must_outlive
        let recursive_bound = {
            let mut components = smallvec![];
            let kind = alias_ty.kind(self.tcx);
            compute_alias_components_recursive(self.tcx, kind, alias_ty, &mut components);
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
            Component::Alias(alias_ty) => self.alias_bound(alias_ty),
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
