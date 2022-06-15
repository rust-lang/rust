use crate::infer::outlives::env::RegionBoundPairs;
use crate::infer::region_constraints::VerifyIfEq;
use crate::infer::{GenericKind, VerifyBound};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::sso::SsoHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, Subst};
use rustc_middle::ty::{self, EarlyBinder, Ty, TyCtxt};

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
        Self { tcx, region_bound_pairs, implicit_region_bound, param_env }
    }

    /// Returns a "verify bound" that encodes what we know about
    /// `generic` and the regions it outlives.
    pub fn generic_bound(&self, generic: GenericKind<'tcx>) -> VerifyBound<'tcx> {
        let mut visited = SsoHashSet::new();
        match generic {
            GenericKind::Param(param_ty) => self.param_bound(param_ty),
            GenericKind::Projection(projection_ty) => {
                self.projection_bound(projection_ty, &mut visited)
            }
        }
    }

    fn type_bound(
        &self,
        ty: Ty<'tcx>,
        visited: &mut SsoHashSet<GenericArg<'tcx>>,
    ) -> VerifyBound<'tcx> {
        match *ty.kind() {
            ty::Param(p) => self.param_bound(p),
            ty::Projection(data) => self.projection_bound(data, visited),
            ty::FnDef(_, substs) => {
                // HACK(eddyb) ignore lifetimes found shallowly in `substs`.
                // This is inconsistent with `ty::Adt` (including all substs),
                // but consistent with previous (accidental) behavior.
                // See https://github.com/rust-lang/rust/issues/70917
                // for further background and discussion.
                let mut bounds = substs
                    .iter()
                    .filter_map(|child| match child.unpack() {
                        GenericArgKind::Type(ty) => Some(self.type_bound(ty, visited)),
                        GenericArgKind::Lifetime(_) => None,
                        GenericArgKind::Const(_) => Some(self.recursive_bound(child, visited)),
                    })
                    .filter(|bound| {
                        // Remove bounds that must hold, since they are not interesting.
                        !bound.must_hold()
                    });

                match (bounds.next(), bounds.next()) {
                    (Some(first), None) => first,
                    (first, second) => VerifyBound::AllBounds(
                        first.into_iter().chain(second).chain(bounds).collect(),
                    ),
                }
            }
            _ => self.recursive_bound(ty.into(), visited),
        }
    }

    fn param_bound(&self, param_ty: ty::ParamTy) -> VerifyBound<'tcx> {
        debug!("param_bound(param_ty={:?})", param_ty);

        // Start with anything like `T: 'a` we can scrape from the
        // environment. If the environment contains something like
        // `for<'a> T: 'a`, then we know that `T` outlives everything.
        let declared_bounds_from_env = self.declared_generic_bounds_from_env(param_ty);
        let mut param_bounds = vec![];
        for declared_bound in declared_bounds_from_env {
            let bound_region = declared_bound.map_bound(|outlives| outlives.1);
            if let Some(region) = bound_region.no_bound_vars() {
                // This is `T: 'a` for some free region `'a`.
                param_bounds.push(VerifyBound::OutlivedBy(region));
            } else {
                // This is `for<'a> T: 'a`. This means that `T` outlives everything! All done here.
                return VerifyBound::AllBounds(vec![]);
            }
        }

        // Add in the default bound of fn body that applies to all in
        // scope type parameters:
        if let Some(r) = self.implicit_region_bound {
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
    pub fn projection_approx_declared_bounds_from_env(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> Vec<ty::Binder<'tcx, ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>> {
        let projection_ty = GenericKind::Projection(projection_ty).to_ty(self.tcx);
        let erased_projection_ty = self.tcx.erase_regions(projection_ty);
        self.declared_generic_bounds_from_env_for_erased_ty(erased_projection_ty)
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

    pub fn projection_bound(
        &self,
        projection_ty: ty::ProjectionTy<'tcx>,
        visited: &mut SsoHashSet<GenericArg<'tcx>>,
    ) -> VerifyBound<'tcx> {
        debug!("projection_bound(projection_ty={:?})", projection_ty);

        let projection_ty_as_ty =
            self.tcx.mk_projection(projection_ty.item_def_id, projection_ty.substs);

        // Search the env for where clauses like `P: 'a`.
        let env_bounds = self
            .projection_approx_declared_bounds_from_env(projection_ty)
            .into_iter()
            .map(|binder| {
                if let Some(ty::OutlivesPredicate(ty, r)) = binder.no_bound_vars() && ty == projection_ty_as_ty {
                    // Micro-optimize if this is an exact match (this
                    // occurs often when there are no region variables
                    // involved).
                    VerifyBound::OutlivedBy(r)
                } else {
                    let verify_if_eq_b = binder.map_bound(|ty::OutlivesPredicate(ty, bound)| VerifyIfEq { ty, bound });
                    VerifyBound::IfEqBound(verify_if_eq_b)
                }
            });

        // Extend with bounds that we can find from the trait.
        let trait_bounds = self
            .projection_declared_bounds_from_trait(projection_ty)
            .map(|r| VerifyBound::OutlivedBy(r));

        // see the extensive comment in projection_must_outlive
        let ty = self.tcx.mk_projection(projection_ty.item_def_id, projection_ty.substs);
        let recursive_bound = self.recursive_bound(ty.into(), visited);

        VerifyBound::AnyBound(env_bounds.chain(trait_bounds).collect()).or(recursive_bound)
    }

    fn recursive_bound(
        &self,
        parent: GenericArg<'tcx>,
        visited: &mut SsoHashSet<GenericArg<'tcx>>,
    ) -> VerifyBound<'tcx> {
        let mut bounds = parent
            .walk_shallow(visited)
            .filter_map(|child| match child.unpack() {
                GenericArgKind::Type(ty) => Some(self.type_bound(ty, visited)),
                GenericArgKind::Lifetime(lt) => {
                    // Ignore late-bound regions.
                    if !lt.is_late_bound() { Some(VerifyBound::OutlivedBy(lt)) } else { None }
                }
                GenericArgKind::Const(_) => Some(self.recursive_bound(child, visited)),
            })
            .filter(|bound| {
                // Remove bounds that must hold, since they are not interesting.
                !bound.must_hold()
            });

        match (bounds.next(), bounds.next()) {
            (Some(first), None) => first,
            (first, second) => {
                VerifyBound::AllBounds(first.into_iter().chain(second).chain(bounds).collect())
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
        param_ty: ty::ParamTy,
    ) -> Vec<ty::Binder<'tcx, ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>> {
        let generic_ty = param_ty.to_ty(self.tcx);
        self.declared_generic_bounds_from_env_for_erased_ty(generic_ty)
    }

    /// Searches the environment to find all bounds that apply to `erased_ty`.
    /// Obviously these must be approximate -- they are in fact both *over* and
    /// and *under* approximated:
    ///
    /// * Over-approximated because we erase regions, so
    /// * Under-approximated because we look for syntactic equality and so for complex types
    ///   like `<T as Foo<fn(&u32, &u32)>>::Item` or whatever we may fail to figure out
    ///   all the subtleties.
    ///
    /// In some cases, such as when `erased_ty` represents a `ty::Param`, however,
    /// the result is precise.
    fn declared_generic_bounds_from_env_for_erased_ty(
        &self,
        erased_ty: Ty<'tcx>,
    ) -> Vec<ty::Binder<'tcx, ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>> {
        let tcx = self.tcx;

        // To start, collect bounds from user environment. Note that
        // parameter environments are already elaborated, so we don't
        // have to worry about that. Comparing using `==` is a bit
        // dubious for projections, but it will work for simple cases
        // like `T` and `T::Item`. It may not work as well for things
        // like `<T as Foo<'a>>::Item`.
        let c_b = self.param_env.caller_bounds();
        let param_bounds = self.collect_outlives_from_predicate_list(erased_ty, c_b.into_iter());

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
                "declared_generic_bounds_from_env_for_erased_ty: region_bound_pair = {:?}",
                (r, p)
            );
            let p_ty = p.to_ty(tcx);
            let erased_p_ty = self.tcx.erase_regions(p_ty);
            (erased_p_ty == erased_ty)
                .then_some(ty::Binder::dummy(ty::OutlivesPredicate(p.to_ty(tcx), r)))
        });

        param_bounds
            .chain(from_region_bound_pairs)
            .inspect(|bound| {
                debug!(
                    "declared_generic_bounds_from_env_for_erased_ty: result predicate = {:?}",
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
            .map(move |r| EarlyBinder(r).subst(tcx, projection_ty.substs))
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
    /// ```compile_fail,E0311
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
    ) -> impl Iterator<Item = ty::Region<'tcx>> {
        let tcx = self.tcx;
        let bounds = tcx.item_bounds(assoc_item_def_id);
        bounds
            .into_iter()
            .filter_map(|p| p.to_opt_type_outlives())
            .filter_map(|p| p.no_bound_vars())
            .map(|b| b.1)
    }

    /// Searches through a predicate list for a predicate `T: 'a`.
    ///
    /// Careful: does not elaborate predicates, and just uses `==`
    /// when comparing `ty` for equality, so `ty` must be something
    /// that does not involve inference variables and where you
    /// otherwise want a precise match.
    fn collect_outlives_from_predicate_list(
        &self,
        erased_ty: Ty<'tcx>,
        predicates: impl Iterator<Item = ty::Predicate<'tcx>>,
    ) -> impl Iterator<Item = ty::Binder<'tcx, ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>>
    {
        let tcx = self.tcx;
        let param_env = self.param_env;
        predicates.filter_map(|p| p.to_opt_type_outlives()).filter(move |outlives_predicate| {
            super::test_type_match::can_match_erased_ty(
                tcx,
                param_env,
                *outlives_predicate,
                erased_ty,
            )
        })
    }
}
