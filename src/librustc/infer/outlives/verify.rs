// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use infer::outlives::env::RegionBoundPairs;
use infer::{GenericKind, VerifyBound};
use traits;
use ty::subst::{Subst, Substs};
use ty::{self, Ty, TyCtxt};

/// The `TypeOutlives` struct has the job of "lowering" a `T: 'a`
/// obligation into a series of `'a: 'b` constraints and "verifys", as
/// described on the module comment. The final constraints are emitted
/// via a "delegate" of type `D` -- this is usually the `infcx`, which
/// accrues them into the `region_obligations` code, but for NLL we
/// use something else.
pub struct VerifyBoundCx<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    region_bound_pairs: &'cx RegionBoundPairs<'tcx>,
    implicit_region_bound: Option<ty::Region<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'cx, 'gcx, 'tcx> VerifyBoundCx<'cx, 'gcx, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        region_bound_pairs: &'cx RegionBoundPairs<'tcx>,
        implicit_region_bound: Option<ty::Region<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        Self {
            tcx,
            region_bound_pairs,
            implicit_region_bound,
            param_env,
        }
    }

    /// Returns a "verify bound" that encodes what we know about
    /// `generic` and the regions it outlives.
    pub fn generic_bound(&self, generic: GenericKind<'tcx>) -> VerifyBound<'tcx> {
        match generic {
            GenericKind::Param(param_ty) => self.param_bound(param_ty),
            GenericKind::Projection(projection_ty) => self.projection_bound(projection_ty),
        }
    }

    fn type_bound(&self, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        match ty.sty {
            ty::Param(p) => self.param_bound(p),
            ty::Projection(data) => self.projection_bound(data),
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

    /// Searches the where clauses in scope for regions that
    /// `projection_ty` is known to outlive. Currently requires an
    /// exact match.
    pub fn projection_declared_bounds(
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

    pub fn projection_bound(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> VerifyBound<'tcx> {
        debug!(
            "projection_bound(projection_ty={:?})",
            projection_ty
        );

        let declared_bounds = self.projection_declared_bounds(projection_ty);

        debug!("projection_bound: declared_bounds = {:?}", declared_bounds);

        // see the extensive comment in projection_must_outlive
        let ty = self.tcx
            .mk_projection(projection_ty.item_def_id, projection_ty.substs);
        let recursive_bound = self.recursive_type_bound(ty);

        VerifyBound::AnyRegion(declared_bounds).or(recursive_bound)
    }

    fn recursive_type_bound(&self, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        let mut bounds = ty.walk_shallow()
            .map(|subty| self.type_bound(subty))
            .collect::<Vec<_>>();

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
        let tcx = self.tcx;

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
            *r = r.subst(self.tcx, projection_ty.substs);
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
        let tcx = self.tcx;
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
