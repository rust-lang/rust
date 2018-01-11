// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code that handles "type-outlives" constraints like `T: 'a`. This
//! is based on the `outlives_components` function defined on the tcx,
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
//! fn bar<T>(a: T, b: impl for<'a> Fn(&'a T));
//! fn foo<T>(x: T) {
//!     bar(x, |y| { ... })
//!          // ^ closure arg
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
//! the type of the closure's first argument would be `&'a ?U`.  We
//! might later infer `?U` to something like `&'b u32`, which would
//! imply that `'b: 'a`.

use hir::def_id::DefId;
use infer::{self, GenericKind, InferCtxt, RegionObligation, SubregionOrigin, VerifyBound};
use traits;
use ty::{self, Ty, TyCtxt, TypeFoldable};
use ty::subst::{Subst, Substs};
use ty::outlives::Component;
use syntax::ast;

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
    /// Registers that the given region obligation must be resolved
    /// from within the scope of `body_id`. These regions are enqueued
    /// and later processed by regionck, when full type information is
    /// available (see `region_obligations` field for more
    /// information).
    pub fn register_region_obligation(
        &self,
        body_id: ast::NodeId,
        obligation: RegionObligation<'tcx>,
    ) {
        debug!(
            "register_region_obligation(body_id={:?}, obligation={:?})",
            body_id,
            obligation
        );

        self.region_obligations
            .borrow_mut()
            .push((body_id, obligation));
    }

    /// Process the region obligations that must be proven (during
    /// `regionck`) for the given `body_id`, given information about
    /// the region bounds in scope and so forth. This function must be
    /// invoked for all relevant body-ids before region inference is
    /// done (or else an assert will fire).
    ///
    /// See the `region_obligations` field of `InferCtxt` for some
    /// comments about how this funtion fits into the overall expected
    /// flow of the the inferencer. The key point is that it is
    /// invoked after all type-inference variables have been bound --
    /// towards the end of regionck. This also ensures that the
    /// region-bound-pairs are available (see comments above regarding
    /// closures).
    ///
    /// # Parameters
    ///
    /// - `region_bound_pairs`: the set of region bounds implied by
    ///   the parameters and where-clauses. In particular, each pair
    ///   `('a, K)` in this list tells us that the bounds in scope
    ///   indicate that `K: 'a`, where `K` is either a generic
    ///   parameter like `T` or a projection like `T::Item`.
    /// - `implicit_region_bound`: if some, this is a region bound
    ///   that is considered to hold for all type parameters (the
    ///   function body).
    /// - `param_env` is the parameter environment for the enclosing function.
    /// - `body_id` is the body-id whose region obligations are being
    ///   processed.
    ///
    /// # Returns
    ///
    /// This function may have to perform normalizations, and hence it
    /// returns an `InferOk` with subobligations that must be
    /// processed.
    pub fn process_registered_region_obligations(
        &self,
        region_bound_pairs: &[(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: ast::NodeId,
    ) {
        assert!(
            !self.in_snapshot.get(),
            "cannot process registered region obligations in a snapshot"
        );

        debug!("process_registered_region_obligations()");

        // pull out the region obligations with the given `body_id` (leaving the rest)
        let mut my_region_obligations = Vec::with_capacity(self.region_obligations.borrow().len());
        {
            let mut r_o = self.region_obligations.borrow_mut();
            for (_, obligation) in r_o.drain_filter(|(ro_body_id, _)| *ro_body_id == body_id) {
                my_region_obligations.push(obligation);
            }
        }

        let outlives =
            TypeOutlives::new(self, region_bound_pairs, implicit_region_bound, param_env);

        for RegionObligation {
            sup_type,
            sub_region,
            cause,
        } in my_region_obligations
        {
            debug!(
                "process_registered_region_obligations: sup_type={:?} sub_region={:?} cause={:?}",
                sup_type,
                sub_region,
                cause
            );

            let origin = SubregionOrigin::from_obligation_cause(
                &cause,
                || infer::RelateParamBound(cause.span, sup_type),
            );

            outlives.type_must_outlive(origin, sup_type, sub_region);
        }
    }

    /// Processes a single ad-hoc region obligation that was not
    /// registered in advance.
    pub fn type_must_outlive(
        &self,
        region_bound_pairs: &[(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        origin: infer::SubregionOrigin<'tcx>,
        ty: Ty<'tcx>,
        region: ty::Region<'tcx>,
    ) {
        let outlives =
            TypeOutlives::new(self, region_bound_pairs, implicit_region_bound, param_env);
        outlives.type_must_outlive(origin, ty, region);
    }
}

#[must_use] // you ought to invoke `into_accrued_obligations` when you are done =)
struct TypeOutlives<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    // See the comments on `process_registered_region_obligations` for the meaning
    // of these fields.
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
    region_bound_pairs: &'cx [(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'cx, 'gcx, 'tcx> TypeOutlives<'cx, 'gcx, 'tcx> {
    fn new(
        infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
        region_bound_pairs: &'cx [(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        Self {
            infcx,
            region_bound_pairs,
            implicit_region_bound,
            param_env,
        }
    }

    /// Adds constraints to inference such that `T: 'a` holds (or
    /// reports an error if it cannot).
    ///
    /// # Parameters
    ///
    /// - `origin`, the reason we need this constraint
    /// - `ty`, the type `T`
    /// - `region`, the region `'a`
    fn type_must_outlive(
        &self,
        origin: infer::SubregionOrigin<'tcx>,
        ty: Ty<'tcx>,
        region: ty::Region<'tcx>,
    ) {
        let ty = self.infcx.resolve_type_vars_if_possible(&ty);

        debug!(
            "type_must_outlive(ty={:?}, region={:?}, origin={:?})",
            ty,
            region,
            origin
        );

        assert!(!ty.has_escaping_regions());

        let components = self.tcx().outlives_components(ty);
        self.components_must_outlive(origin, components, region);
    }

    fn tcx(&self) -> TyCtxt<'cx, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn components_must_outlive(
        &self,
        origin: infer::SubregionOrigin<'tcx>,
        components: Vec<Component<'tcx>>,
        region: ty::Region<'tcx>,
    ) {
        for component in components {
            let origin = origin.clone();
            match component {
                Component::Region(region1) => {
                    self.infcx.sub_regions(origin, region, region1);
                }
                Component::Param(param_ty) => {
                    self.param_ty_must_outlive(origin, region, param_ty);
                }
                Component::Projection(projection_ty) => {
                    self.projection_must_outlive(origin, region, projection_ty);
                }
                Component::EscapingProjection(subcomponents) => {
                    self.components_must_outlive(origin, subcomponents, region);
                }
                Component::UnresolvedInferenceVariable(v) => {
                    // ignore this, we presume it will yield an error
                    // later, since if a type variable is not resolved by
                    // this point it never will be
                    self.infcx.tcx.sess.delay_span_bug(
                        origin.span(),
                        &format!("unresolved inference variable in outlives: {:?}", v),
                    );
                }
            }
        }
    }

    fn param_ty_must_outlive(
        &self,
        origin: infer::SubregionOrigin<'tcx>,
        region: ty::Region<'tcx>,
        param_ty: ty::ParamTy,
    ) {
        debug!(
            "param_ty_must_outlive(region={:?}, param_ty={:?}, origin={:?})",
            region,
            param_ty,
            origin
        );

        let verify_bound = self.param_bound(param_ty);
        let generic = GenericKind::Param(param_ty);
        self.infcx
            .verify_generic_bound(origin, generic, region, verify_bound);
    }

    fn projection_must_outlive(
        &self,
        origin: infer::SubregionOrigin<'tcx>,
        region: ty::Region<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) {
        debug!(
            "projection_must_outlive(region={:?}, projection_ty={:?}, origin={:?})",
            region,
            projection_ty,
            origin
        );

        // This case is thorny for inference. The fundamental problem is
        // that there are many cases where we have choice, and inference
        // doesn't like choice (the current region inference in
        // particular). :) First off, we have to choose between using the
        // OutlivesProjectionEnv, OutlivesProjectionTraitDef, and
        // OutlivesProjectionComponent rules, any one of which is
        // sufficient.  If there are no inference variables involved, it's
        // not hard to pick the right rule, but if there are, we're in a
        // bit of a catch 22: if we picked which rule we were going to
        // use, we could add constraints to the region inference graph
        // that make it apply, but if we don't add those constraints, the
        // rule might not apply (but another rule might). For now, we err
        // on the side of adding too few edges into the graph.

        // Compute the bounds we can derive from the environment or trait
        // definition.  We know that the projection outlives all the
        // regions in this list.
        let env_bounds = self.projection_declared_bounds(projection_ty);

        debug!("projection_must_outlive: env_bounds={:?}", env_bounds);

        // If we know that the projection outlives 'static, then we're
        // done here.
        if env_bounds.contains(&&ty::ReStatic) {
            debug!("projection_must_outlive: 'static as declared bound");
            return;
        }

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
        let needs_infer = projection_ty.needs_infer();
        if env_bounds.is_empty() && needs_infer {
            debug!("projection_must_outlive: no declared bounds");

            for component_ty in projection_ty.substs.types() {
                self.type_must_outlive(origin.clone(), component_ty, region);
            }

            for r in projection_ty.substs.regions() {
                self.infcx.sub_regions(origin.clone(), region, r);
            }

            return;
        }

        // If we find that there is a unique declared bound `'b`, and this bound
        // appears in the trait reference, then the best action is to require that `'b:'r`,
        // so do that. This is best no matter what rule we use:
        //
        // - OutlivesProjectionEnv or OutlivesProjectionTraitDef: these would translate to
        // the requirement that `'b:'r`
        // - OutlivesProjectionComponent: this would require `'b:'r` in addition to
        // other conditions
        if !env_bounds.is_empty() && env_bounds[1..].iter().all(|b| *b == env_bounds[0]) {
            let unique_bound = env_bounds[0];
            debug!(
                "projection_must_outlive: unique declared bound = {:?}",
                unique_bound
            );
            if projection_ty
                .substs
                .regions()
                .any(|r| env_bounds.contains(&r))
            {
                debug!("projection_must_outlive: unique declared bound appears in trait ref");
                self.infcx.sub_regions(origin.clone(), region, unique_bound);
                return;
            }
        }

        // Fallback to verifying after the fact that there exists a
        // declared bound, or that all the components appearing in the
        // projection outlive; in some cases, this may add insufficient
        // edges into the inference graph, leading to inference failures
        // even though a satisfactory solution exists.
        let verify_bound = self.projection_bound(env_bounds, projection_ty);
        let generic = GenericKind::Projection(projection_ty);
        self.infcx
            .verify_generic_bound(origin, generic.clone(), region, verify_bound);
    }

    fn type_bound(&self, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        match ty.sty {
            ty::TyParam(p) => self.param_bound(p),
            ty::TyProjection(data) => {
                let declared_bounds = self.projection_declared_bounds(data);
                self.projection_bound(declared_bounds, data)
            }
            _ => self.recursive_type_bound(ty),
        }
    }

    fn param_bound(&self, param_ty: ty::ParamTy) -> VerifyBound<'tcx> {
        debug!("param_bound(param_ty={:?})", param_ty);

        let mut param_bounds = self.declared_generic_bounds_from_env(GenericKind::Param(param_ty));

        // Add in the default bound of fn body that applies to all in
        // scope type parameters:
        param_bounds.extend(self.implicit_region_bound);

        VerifyBound::AnyRegion(param_bounds)
    }

    fn projection_declared_bounds(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> Vec<ty::Region<'tcx>> {
        // First assemble bounds from where clauses and traits.

        let mut declared_bounds =
            self.declared_generic_bounds_from_env(GenericKind::Projection(projection_ty));

        declared_bounds
            .extend_from_slice(&self.declared_projection_bounds_from_trait(projection_ty));

        declared_bounds
    }

    fn projection_bound(
        &self,
        declared_bounds: Vec<ty::Region<'tcx>>,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> VerifyBound<'tcx> {
        debug!(
            "projection_bound(declared_bounds={:?}, projection_ty={:?})",
            declared_bounds,
            projection_ty
        );

        // see the extensive comment in projection_must_outlive
        let ty = self.infcx
            .tcx
            .mk_projection(projection_ty.item_def_id, projection_ty.substs);
        let recursive_bound = self.recursive_type_bound(ty);

        VerifyBound::AnyRegion(declared_bounds).or(recursive_bound)
    }

    fn recursive_type_bound(&self, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        let mut bounds = vec![];

        for subty in ty.walk_shallow() {
            bounds.push(self.type_bound(subty));
        }

        let mut regions = ty.regions();
        regions.retain(|r| !r.is_late_bound()); // ignore late-bound regions
        bounds.push(VerifyBound::AllRegions(regions));

        // remove bounds that must hold, since they are not interesting
        bounds.retain(|b| !b.must_hold());

        if bounds.len() == 1 {
            bounds.pop().unwrap()
        } else {
            VerifyBound::AllBounds(bounds)
        }
    }

    fn declared_generic_bounds_from_env(
        &self,
        generic: GenericKind<'tcx>,
    ) -> Vec<ty::Region<'tcx>> {
        let tcx = self.tcx();

        // To start, collect bounds from user environment. Note that
        // parameter environments are already elaborated, so we don't
        // have to worry about that. Comparing using `==` is a bit
        // dubious for projections, but it will work for simple cases
        // like `T` and `T::Item`. It may not work as well for things
        // like `<T as Foo<'a>>::Item`.
        let generic_ty = generic.to_ty(tcx);
        let c_b = self.param_env.caller_bounds;
        let mut param_bounds = self.collect_outlives_from_predicate_list(generic_ty, c_b);

        // Next, collect regions we scraped from the well-formedness
        // constraints in the fn signature. To do that, we walk the list
        // of known relations from the fn ctxt.
        //
        // This is crucial because otherwise code like this fails:
        //
        //     fn foo<'a, A>(x: &'a A) { x.bar() }
        //
        // The problem is that the type of `x` is `&'a A`. To be
        // well-formed, then, A must be lower-generic by `'a`, but we
        // don't know that this holds from first principles.
        for &(r, p) in self.region_bound_pairs {
            debug!("generic={:?} p={:?}", generic, p);
            if generic == p {
                param_bounds.push(r);
            }
        }

        param_bounds
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
    /// then this function would return `'x`. This is subject to the
    /// limitations around higher-ranked bounds described in
    /// `region_bounds_declared_on_associated_item`.
    fn declared_projection_bounds_from_trait(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> Vec<ty::Region<'tcx>> {
        debug!("projection_bounds(projection_ty={:?})", projection_ty);
        let mut bounds = self.region_bounds_declared_on_associated_item(projection_ty.item_def_id);
        for r in &mut bounds {
            *r = r.subst(self.tcx(), projection_ty.substs);
        }
        bounds
    }

    /// Given the def-id of an associated item, returns any region
    /// bounds attached to that associated item from the trait definition.
    ///
    /// For example:
    ///
    /// ```rust
    /// trait Foo<'a> {
    ///     type Bar: 'a;
    /// }
    /// ```
    ///
    /// If we were given the def-id of `Foo::Bar`, we would return
    /// `'a`. You could then apply the substitutions from the
    /// projection to convert this into your namespace. This also
    /// works if the user writes `where <Self as Foo<'a>>::Bar: 'a` on
    /// the trait. In fact, it works by searching for just such a
    /// where-clause.
    ///
    /// It will not, however, work for higher-ranked bounds like:
    ///
    /// ```rust
    /// trait Foo<'a, 'b>
    /// where for<'x> <Self as Foo<'x, 'b>>::Bar: 'x
    /// {
    ///     type Bar;
    /// }
    /// ```
    ///
    /// This is for simplicity, and because we are not really smart
    /// enough to cope with such bounds anywhere.
    fn region_bounds_declared_on_associated_item(
        &self,
        assoc_item_def_id: DefId,
    ) -> Vec<ty::Region<'tcx>> {
        let tcx = self.tcx();
        let assoc_item = tcx.associated_item(assoc_item_def_id);
        let trait_def_id = assoc_item.container.assert_trait();
        let trait_predicates = tcx.predicates_of(trait_def_id);
        let identity_substs = Substs::identity_for_item(tcx, assoc_item_def_id);
        let identity_proj = tcx.mk_projection(assoc_item_def_id, identity_substs);
        self.collect_outlives_from_predicate_list(
            identity_proj,
            traits::elaborate_predicates(tcx, trait_predicates.predicates),
        )
    }

    /// Searches through a predicate list for a predicate `T: 'a`.
    ///
    /// Careful: does not elaborate predicates, and just uses `==`
    /// when comparing `ty` for equality, so `ty` must be something
    /// that does not involve inference variables and where you
    /// otherwise want a precise match.
    fn collect_outlives_from_predicate_list<I, P>(
        &self,
        ty: Ty<'tcx>,
        predicates: I,
    ) -> Vec<ty::Region<'tcx>>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<ty::Predicate<'tcx>>,
    {
        predicates
            .into_iter()
            .filter_map(|p| p.as_ref().to_opt_type_outlives())
            .filter_map(|p| p.no_late_bound_regions())
            .filter(|p| p.0 == ty)
            .map(|p| p.1)
            .collect()
    }
}
