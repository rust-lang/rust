use crate::hir::def_id::DefId;
use crate::infer::outlives::env::RegionBoundPairs;
use crate::infer::{GenericKind, VerifyBound};
use crate::traits;
use crate::ty::subst::{Subst, InternalSubsts};
use crate::ty::{self, Ty, TyCtxt};
use crate::util::captures::Captures;

/// The `TypeOutlives` struct has the job of "lowering" a `T: 'a`
/// obligation into a series of `'a: 'b` constraints and "verifys", as
/// described on the module comment. The final constraints are emitted
/// via a "delegate" of type `D` -- this is usually the `infcx`, which
/// accrues them into the `region_obligations` code, but for NLL we
/// use something else.
pub struct VerifyBoundCx<'cx, 'tcx> {
    tcx: TyCtxt<'tcx>,
    region_bound_pairs: &'cx RegionBoundPairs<'tcx>,
    implicit_region_bound: Option<ty::Region<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'cx, 'tcx> VerifyBoundCx<'cx, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
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

        // Start with anything like `T: 'a` we can scrape from the
        // environment
        let param_bounds = self.declared_generic_bounds_from_env(GenericKind::Param(param_ty))
            .into_iter()
            .map(|outlives| outlives.1);

        // Add in the default bound of fn body that applies to all in
        // scope type parameters:
        let param_bounds = param_bounds.chain(self.implicit_region_bound);

        VerifyBound::AnyBound(param_bounds.map(|r| VerifyBound::OutlivedBy(r)).collect())
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
    pub fn projection_approx_declared_bounds_from_env(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> Vec<ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>> {
        let projection_ty = GenericKind::Projection(projection_ty).to_ty(self.tcx);
        let erased_projection_ty = self.tcx.erase_regions(&projection_ty);
        self.declared_generic_bounds_from_env_with_compare_fn(|ty| {
            if let ty::Projection(..) = ty.sty {
                let erased_ty = self.tcx.erase_regions(&ty);
                erased_ty == erased_projection_ty
            } else {
                false
            }
        })
    }

    /// Searches the where-clauses in scope for regions that
    /// `projection_ty` is known to outlive. Currently requires an
    /// exact match.
    pub fn projection_declared_bounds_from_trait(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> impl Iterator<Item = ty::Region<'tcx>> + 'cx + Captures<'tcx> {
        self.declared_projection_bounds_from_trait(projection_ty)
    }

    pub fn projection_bound(&self, projection_ty: ty::ProjectionTy<'tcx>) -> VerifyBound<'tcx> {
        debug!("projection_bound(projection_ty={:?})", projection_ty);

        let projection_ty_as_ty =
            self.tcx.mk_projection(projection_ty.item_def_id, projection_ty.substs);

        // Search the env for where clauses like `P: 'a`.
        let env_bounds = self.projection_approx_declared_bounds_from_env(projection_ty)
            .into_iter()
            .map(|ty::OutlivesPredicate(ty, r)| {
                let vb = VerifyBound::OutlivedBy(r);
                if ty == projection_ty_as_ty {
                    // Micro-optimize if this is an exact match (this
                    // occurs often when there are no region variables
                    // involved).
                    vb
                } else {
                    VerifyBound::IfEq(ty, Box::new(vb))
                }
            });

        // Extend with bounds that we can find from the trait.
        let trait_bounds = self.projection_declared_bounds_from_trait(projection_ty)
            .into_iter()
            .map(|r| VerifyBound::OutlivedBy(r));

        // see the extensive comment in projection_must_outlive
        let ty = self.tcx
            .mk_projection(projection_ty.item_def_id, projection_ty.substs);
        let recursive_bound = self.recursive_type_bound(ty);

        VerifyBound::AnyBound(env_bounds.chain(trait_bounds).collect()).or(recursive_bound)
    }

    fn recursive_type_bound(&self, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        let mut bounds = ty.walk_shallow()
            .map(|subty| self.type_bound(subty))
            .collect::<Vec<_>>();

        let mut regions = smallvec![];
        ty.push_regions(&mut regions);
        regions.retain(|r| !r.is_late_bound()); // ignore late-bound regions
        bounds.push(VerifyBound::AllBounds(
            regions
                .into_iter()
                .map(|r| VerifyBound::OutlivedBy(r))
                .collect(),
        ));

        // remove bounds that must hold, since they are not interesting
        bounds.retain(|b| !b.must_hold());

        if bounds.len() == 1 {
            bounds.pop().unwrap()
        } else {
            VerifyBound::AllBounds(bounds)
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
        generic: GenericKind<'tcx>,
    ) -> Vec<ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>> {
        let generic_ty = generic.to_ty(self.tcx);
        self.declared_generic_bounds_from_env_with_compare_fn(|ty| ty == generic_ty)
    }

    fn declared_generic_bounds_from_env_with_compare_fn(
        &self,
        compare_ty: impl Fn(Ty<'tcx>) -> bool,
    ) -> Vec<ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>> {
        let tcx = self.tcx;

        // To start, collect bounds from user environment. Note that
        // parameter environments are already elaborated, so we don't
        // have to worry about that. Comparing using `==` is a bit
        // dubious for projections, but it will work for simple cases
        // like `T` and `T::Item`. It may not work as well for things
        // like `<T as Foo<'a>>::Item`.
        let c_b = self.param_env.caller_bounds;
        let param_bounds = self.collect_outlives_from_predicate_list(&compare_ty, c_b);

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
        let from_region_bound_pairs = self.region_bound_pairs.iter().filter_map(|&(r, p)| {
            debug!(
                "declared_generic_bounds_from_env_with_compare_fn: region_bound_pair = {:?}",
                (r, p)
            );
            let p_ty = p.to_ty(tcx);
            if compare_ty(p_ty) {
                Some(ty::OutlivesPredicate(p_ty, r))
            } else {
                None
            }
        });

        param_bounds
            .chain(from_region_bound_pairs)
            .inspect(|bound| {
                debug!(
                    "declared_generic_bounds_from_env_with_compare_fn: result predicate = {:?}",
                    bound
                )
            })
            .collect()
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
    ) -> impl Iterator<Item = ty::Region<'tcx>> + 'cx + Captures<'tcx> {
        debug!("projection_bounds(projection_ty={:?})", projection_ty);
        let tcx = self.tcx;
        self.region_bounds_declared_on_associated_item(projection_ty.item_def_id)
            .map(move |r| r.subst(tcx, projection_ty.substs))
    }

    /// Given the `DefId` of an associated item, returns any region
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
    /// If we were given the `DefId` of `Foo::Bar`, we would return
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
    ) -> impl Iterator<Item = ty::Region<'tcx>> + 'cx + Captures<'tcx> {
        let tcx = self.tcx;
        let assoc_item = tcx.associated_item(assoc_item_def_id);
        let trait_def_id = assoc_item.container.assert_trait();
        let trait_predicates = tcx.predicates_of(trait_def_id).predicates
            .iter()
            .map(|(p, _)| *p)
            .collect();
        let identity_substs = InternalSubsts::identity_for_item(tcx, assoc_item_def_id);
        let identity_proj = tcx.mk_projection(assoc_item_def_id, identity_substs);
        self.collect_outlives_from_predicate_list(
            move |ty| ty == identity_proj,
            traits::elaborate_predicates(tcx, trait_predicates),
        ).map(|b| b.1)
    }

    /// Searches through a predicate list for a predicate `T: 'a`.
    ///
    /// Careful: does not elaborate predicates, and just uses `==`
    /// when comparing `ty` for equality, so `ty` must be something
    /// that does not involve inference variables and where you
    /// otherwise want a precise match.
    fn collect_outlives_from_predicate_list(
        &self,
        compare_ty: impl Fn(Ty<'tcx>) -> bool,
        predicates: impl IntoIterator<Item = impl AsRef<ty::Predicate<'tcx>>>,
    ) -> impl Iterator<Item = ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>> {
        predicates
            .into_iter()
            .filter_map(|p| p.as_ref().to_opt_type_outlives())
            .filter_map(|p| p.no_bound_vars())
            .filter(move |p| compare_ty(p.0))
    }
}
