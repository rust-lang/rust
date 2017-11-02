//! Temporary holding spot for some code I want to factor out.

use rustc::traits::{self, ObligationCause, ObligationCauseCode, PredicateObligations};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::infer::{self, GenericKind, InferCtxt, InferOk, VerifyBound};
use rustc::ty::subst::Subst;
use rustc::ty::outlives::Component;
use syntax::ast;
use syntax_pos::Span;

pub struct RegionckOutlives<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    // Context provided by the caller:
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
    region_bound_pairs: &'cx [(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: ast::NodeId,

    // Obligations that we accrue as we go:
    obligations: PredicateObligations<'tcx>,
}

impl<'cx, 'gcx, 'tcx> RegionckOutlives<'cx, 'gcx, 'tcx> {
    pub fn new(
        infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
        region_bound_pairs: &'cx [(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: ast::NodeId,
    ) -> Self {
        Self {
            infcx,
            region_bound_pairs,
            implicit_region_bound,
            param_env,
            body_id,
            obligations: vec![],
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
    pub fn type_must_outlive(
        mut self,
        origin: infer::SubregionOrigin<'tcx>,
        ty: Ty<'tcx>,
        region: ty::Region<'tcx>,
    ) -> InferOk<'tcx, ()> {
        self.type_must_outlive_pushing_obligations(origin, ty, region);
        InferOk {
            value: (),
            obligations: self.obligations,
        }
    }

    /// Internal helper: ensure that `ty_must_outlive` and push obligations onto
    /// our internal vector.
    fn type_must_outlive_pushing_obligations(
        &mut self,
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
        &mut self,
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
        &mut self,
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
        &mut self,
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
        let env_bounds = self.projection_declared_bounds(origin.span(), projection_ty);

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
                self.type_must_outlive_pushing_obligations(origin.clone(), component_ty, region);
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
        let verify_bound = self.projection_bound(origin.span(), env_bounds, projection_ty);
        let generic = GenericKind::Projection(projection_ty);
        self.infcx
            .verify_generic_bound(origin, generic.clone(), region, verify_bound);
    }

    fn type_bound(&mut self, span: Span, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        match ty.sty {
            ty::TyParam(p) => self.param_bound(p),
            ty::TyProjection(data) => {
                let declared_bounds = self.projection_declared_bounds(span, data);
                self.projection_bound(span, declared_bounds, data)
            }
            _ => self.recursive_type_bound(span, ty),
        }
    }

    fn param_bound(&mut self, param_ty: ty::ParamTy) -> VerifyBound<'tcx> {
        debug!("param_bound(param_ty={:?})", param_ty);

        let mut param_bounds = self.declared_generic_bounds_from_env(GenericKind::Param(param_ty));

        // Add in the default bound of fn body that applies to all in
        // scope type parameters:
        param_bounds.extend(self.implicit_region_bound);

        VerifyBound::AnyRegion(param_bounds)
    }

    fn projection_declared_bounds(
        &mut self,
        span: Span,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> Vec<ty::Region<'tcx>> {
        // First assemble bounds from where clauses and traits.

        let mut declared_bounds =
            self.declared_generic_bounds_from_env(GenericKind::Projection(projection_ty));

        declared_bounds
            .extend_from_slice(&mut self.declared_projection_bounds_from_trait(span, projection_ty));

        declared_bounds
    }

    fn projection_bound(
        &mut self,
        span: Span,
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
        let recursive_bound = self.recursive_type_bound(span, ty);

        VerifyBound::AnyRegion(declared_bounds).or(recursive_bound)
    }

    fn recursive_type_bound(&mut self, span: Span, ty: Ty<'tcx>) -> VerifyBound<'tcx> {
        let mut bounds = vec![];

        for subty in ty.walk_shallow() {
            bounds.push(self.type_bound(span, subty));
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
        &mut self,
        generic: GenericKind<'tcx>,
    ) -> Vec<ty::Region<'tcx>> {
        let tcx = self.tcx();

        // To start, collect bounds from user:
        let mut param_bounds =
            tcx.required_region_bounds(generic.to_ty(tcx), self.param_env.caller_bounds.to_vec());

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

    fn declared_projection_bounds_from_trait(
        &mut self,
        span: Span,
        projection_ty: ty::ProjectionTy<'tcx>,
    ) -> Vec<ty::Region<'tcx>> {
        debug!("projection_bounds(projection_ty={:?})", projection_ty);
        let ty = self.tcx()
            .mk_projection(projection_ty.item_def_id, projection_ty.substs);

        // Say we have a projection `<T as SomeTrait<'a>>::SomeType`. We are interested
        // in looking for a trait definition like:
        //
        // ```
        // trait SomeTrait<'a> {
        //     type SomeType : 'a;
        // }
        // ```
        //
        // we can thus deduce that `<T as SomeTrait<'a>>::SomeType : 'a`.
        let trait_predicates = self.tcx()
            .predicates_of(projection_ty.trait_ref(self.tcx()).def_id);
        assert_eq!(trait_predicates.parent, None);
        let predicates = trait_predicates.predicates.as_slice().to_vec();
        traits::elaborate_predicates(self.tcx(), predicates)
            .filter_map(|predicate| {
                // we're only interesting in `T : 'a` style predicates:
                let outlives = match predicate {
                    ty::Predicate::TypeOutlives(data) => data,
                    _ => {
                        return None;
                    }
                };

                debug!("projection_bounds: outlives={:?} (1)", outlives);

                // apply the substitutions (and normalize any projected types)
                let outlives = outlives.subst(self.tcx(), projection_ty.substs);
                let outlives = self.infcx.partially_normalize_associated_types_in(
                    span,
                    self.body_id,
                    self.param_env,
                    &outlives,
                );
                let outlives = self.register_infer_ok_obligations(outlives);

                debug!("projection_bounds: outlives={:?} (2)", outlives);

                let region_result = self.infcx
                    .commit_if_ok(|_| {
                        let (outlives, _) = self.infcx.replace_late_bound_regions_with_fresh_var(
                            span,
                            infer::AssocTypeProjection(projection_ty.item_def_id),
                            &outlives,
                        );

                        debug!("projection_bounds: outlives={:?} (3)", outlives);

                        // check whether this predicate applies to our current projection
                        let cause = ObligationCause::new(
                            span,
                            self.body_id,
                            ObligationCauseCode::MiscObligation,
                        );
                        match self.infcx.at(&cause, self.param_env).eq(outlives.0, ty) {
                            Ok(ok) => Ok((ok, outlives.1)),
                            Err(_) => Err(()),
                        }
                    })
                    .map(|(ok, result)| {
                        self.register_infer_ok_obligations(ok);
                        result
                    });

                debug!("projection_bounds: region_result={:?}", region_result);

                region_result.ok()
            })
            .collect()
    }

    fn register_infer_ok_obligations<T>(&mut self, infer_ok: InferOk<'tcx, T>) -> T {
        let InferOk { value, obligations } = infer_ok;
        self.obligations.extend(obligations);
        value
    }
}
