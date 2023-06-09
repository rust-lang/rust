use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::OpaqueTyOrigin;
use rustc_infer::infer::InferCtxt;
use rustc_infer::infer::TyCtxtInferExt as _;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_middle::traits::DefiningAnchor;
use rustc_middle::ty::subst::{GenericArgKind, InternalSubsts};
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, OpaqueHiddenType, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::ObligationCtxt;

use crate::session_diagnostics::NonGenericOpaqueTypeParam;

use super::RegionInferenceContext;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Resolve any opaque types that were encountered while borrow checking
    /// this item. This is then used to get the type in the `type_of` query.
    ///
    /// For example consider `fn f<'a>(x: &'a i32) -> impl Sized + 'a { x }`.
    /// This is lowered to give HIR something like
    ///
    /// type f<'a>::_Return<'_a> = impl Sized + '_a;
    /// fn f<'a>(x: &'a i32) -> f<'static>::_Return<'a> { x }
    ///
    /// When checking the return type record the type from the return and the
    /// type used in the return value. In this case they might be `_Return<'1>`
    /// and `&'2 i32` respectively.
    ///
    /// Once we to this method, we have completed region inference and want to
    /// call `infer_opaque_definition_from_instantiation` to get the inferred
    /// type of `_Return<'_a>`. `infer_opaque_definition_from_instantiation`
    /// compares lifetimes directly, so we need to map the inference variables
    /// back to concrete lifetimes: `'static`, `ReEarlyBound` or `ReFree`.
    ///
    /// First we map all the lifetimes in the concrete type to an equal
    /// universal region that occurs in the concrete type's substs, in this case
    /// this would result in `&'1 i32`. We only consider regions in the substs
    /// in case there is an equal region that does not. For example, this should
    /// be allowed:
    /// `fn f<'a: 'b, 'b: 'a>(x: *mut &'b i32) -> impl Sized + 'a { x }`
    ///
    /// Then we map the regions in both the type and the subst to their
    /// `external_name` giving `concrete_type = &'a i32`,
    /// `substs = ['static, 'a]`. This will then allow
    /// `infer_opaque_definition_from_instantiation` to determine that
    /// `_Return<'_a> = &'_a i32`.
    ///
    /// There's a slight complication around closures. Given
    /// `fn f<'a: 'a>() { || {} }` the closure's type is something like
    /// `f::<'a>::{{closure}}`. The region parameter from f is essentially
    /// ignored by type checking so ends up being inferred to an empty region.
    /// Calling `universal_upper_bound` for such a region gives `fr_fn_body`,
    /// which has no `external_name` in which case we use `'empty` as the
    /// region to pass to `infer_opaque_definition_from_instantiation`.
    #[instrument(level = "debug", skip(self, infcx), ret)]
    pub(crate) fn infer_opaque_types(
        &self,
        infcx: &InferCtxt<'tcx>,
        opaque_ty_decls: FxIndexMap<OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>>,
    ) -> FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>> {
        let mut result: FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>> = FxIndexMap::default();

        let member_constraints: FxIndexMap<_, _> = self
            .member_constraints
            .all_indices()
            .map(|ci| (self.member_constraints[ci].key, ci))
            .collect();
        debug!(?member_constraints);

        for (opaque_type_key, concrete_type) in opaque_ty_decls {
            let substs = opaque_type_key.substs;
            debug!(?concrete_type, ?substs);

            let mut subst_regions = vec![self.universal_regions.fr_static];

            let to_universal_region = |vid, subst_regions: &mut Vec<_>| {
                trace!(?vid);
                let scc = self.constraint_sccs.scc(vid);
                trace!(?scc);
                match self.scc_values.universal_regions_outlived_by(scc).find_map(|lb| {
                    self.eval_equal(vid, lb).then_some(self.definitions[lb].external_name?)
                }) {
                    Some(region) => {
                        let vid = self.universal_regions.to_region_vid(region);
                        subst_regions.push(vid);
                        region
                    }
                    None => {
                        subst_regions.push(vid);
                        ty::Region::new_error_with_message(
                            infcx.tcx,
                            concrete_type.span,
                            "opaque type with non-universal region substs",
                        )
                    }
                }
            };

            // Start by inserting universal regions from the member_constraint choice regions.
            // This will ensure they get precedence when folding the regions in the concrete type.
            if let Some(&ci) = member_constraints.get(&opaque_type_key) {
                for &vid in self.member_constraints.choice_regions(ci) {
                    to_universal_region(vid, &mut subst_regions);
                }
            }
            debug!(?subst_regions);

            // Next, insert universal regions from substs, so we can translate regions that appear
            // in them but are not subject to member constraints, for instance closure substs.
            let universal_substs = infcx.tcx.fold_regions(substs, |region, _| {
                if let ty::RePlaceholder(..) = region.kind() {
                    // Higher kinded regions don't need remapping, they don't refer to anything outside of this the substs.
                    return region;
                }
                let vid = self.to_region_vid(region);
                to_universal_region(vid, &mut subst_regions)
            });
            debug!(?universal_substs);
            debug!(?subst_regions);

            // Deduplicate the set of regions while keeping the chosen order.
            let subst_regions = subst_regions.into_iter().collect::<FxIndexSet<_>>();
            debug!(?subst_regions);

            let universal_concrete_type =
                infcx.tcx.fold_regions(concrete_type, |region, _| match *region {
                    ty::ReVar(vid) => subst_regions
                        .iter()
                        .find(|ur_vid| self.eval_equal(vid, **ur_vid))
                        .and_then(|ur_vid| self.definitions[*ur_vid].external_name)
                        .unwrap_or(infcx.tcx.lifetimes.re_erased),
                    _ => region,
                });
            debug!(?universal_concrete_type);

            let opaque_type_key =
                OpaqueTypeKey { def_id: opaque_type_key.def_id, substs: universal_substs };
            let ty = infcx.infer_opaque_definition_from_instantiation(
                opaque_type_key,
                universal_concrete_type,
            );
            // Sometimes two opaque types are the same only after we remap the generic parameters
            // back to the opaque type definition. E.g. we may have `OpaqueType<X, Y>` mapped to `(X, Y)`
            // and `OpaqueType<Y, X>` mapped to `(Y, X)`, and those are the same, but we only know that
            // once we convert the generic parameters to those of the opaque type.
            if let Some(prev) = result.get_mut(&opaque_type_key.def_id) {
                if prev.ty != ty {
                    let guar = ty.error_reported().err().unwrap_or_else(|| {
                        prev.report_mismatch(
                            &OpaqueHiddenType { ty, span: concrete_type.span },
                            opaque_type_key.def_id,
                            infcx.tcx,
                        )
                        .emit()
                    });
                    prev.ty = infcx.tcx.ty_error(guar);
                }
                // Pick a better span if there is one.
                // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
                prev.span = prev.span.substitute_dummy(concrete_type.span);
            } else {
                result.insert(
                    opaque_type_key.def_id,
                    OpaqueHiddenType { ty, span: concrete_type.span },
                );
            }
        }
        result
    }

    /// Map the regions in the type to named regions. This is similar to what
    /// `infer_opaque_types` does, but can infer any universal region, not only
    /// ones from the substs for the opaque type. It also doesn't double check
    /// that the regions produced are in fact equal to the named region they are
    /// replaced with. This is fine because this function is only to improve the
    /// region names in error messages.
    pub(crate) fn name_regions<T>(&self, tcx: TyCtxt<'tcx>, ty: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        tcx.fold_regions(ty, |region, _| match *region {
            ty::ReVar(vid) => {
                // Find something that we can name
                let upper_bound = self.approx_universal_upper_bound(vid);
                let upper_bound = &self.definitions[upper_bound];
                match upper_bound.external_name {
                    Some(reg) => reg,
                    None => {
                        // Nothing exact found, so we pick the first one that we find.
                        let scc = self.constraint_sccs.scc(vid);
                        for vid in self.rev_scc_graph.as_ref().unwrap().upper_bounds(scc) {
                            match self.definitions[vid].external_name {
                                None => {}
                                Some(region) if region.is_static() => {}
                                Some(region) => return region,
                            }
                        }
                        region
                    }
                }
            }
            _ => region,
        })
    }
}

pub trait InferCtxtExt<'tcx> {
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
    ) -> Ty<'tcx>;
}

impl<'tcx> InferCtxtExt<'tcx> for InferCtxt<'tcx> {
    /// Given the fully resolved, instantiated type for an opaque
    /// type, i.e., the value of an inference variable like C1 or C2
    /// (*), computes the "definition type" for an opaque type
    /// definition -- that is, the inferred value of `Foo1<'x>` or
    /// `Foo2<'x>` that we would conceptually use in its definition:
    /// ```ignore (illustrative)
    /// type Foo1<'x> = impl Bar<'x> = AAA;  // <-- this type AAA
    /// type Foo2<'x> = impl Bar<'x> = BBB;  // <-- or this type BBB
    /// fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
    /// ```
    /// Note that these values are defined in terms of a distinct set of
    /// generic parameters (`'x` instead of `'a`) from C1 or C2. The main
    /// purpose of this function is to do that translation.
    ///
    /// (*) C1 and C2 were introduced in the comments on
    /// `register_member_constraints`. Read that comment for more context.
    ///
    /// # Parameters
    ///
    /// - `def_id`, the `impl Trait` type
    /// - `substs`, the substs used to instantiate this opaque type
    /// - `instantiated_ty`, the inferred type C1 -- fully resolved, lifted version of
    ///   `opaque_defn.concrete_ty`
    #[instrument(level = "debug", skip(self))]
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
    ) -> Ty<'tcx> {
        if let Some(e) = self.tainted_by_errors() {
            return self.tcx.ty_error(e);
        }

        if let Err(guar) =
            check_opaque_type_parameter_valid(self.tcx, opaque_type_key, instantiated_ty.span)
        {
            return self.tcx.ty_error(guar);
        }

        let definition_ty = instantiated_ty
            .remap_generic_params_to_declaration_params(opaque_type_key, self.tcx, false)
            .ty;

        // `definition_ty` does not live in of the current inference context,
        // so lets make sure that we don't accidentally misuse our current `infcx`.
        match check_opaque_type_well_formed(
            self.tcx,
            self.next_trait_solver(),
            opaque_type_key.def_id,
            instantiated_ty.span,
            definition_ty,
        ) {
            Ok(hidden_ty) => hidden_ty,
            Err(guar) => self.tcx.ty_error(guar),
        }
    }
}

/// This logic duplicates most of `check_opaque_meets_bounds`.
/// FIXME(oli-obk): Also do region checks here and then consider removing
/// `check_opaque_meets_bounds` entirely.
fn check_opaque_type_well_formed<'tcx>(
    tcx: TyCtxt<'tcx>,
    next_trait_solver: bool,
    def_id: LocalDefId,
    definition_span: Span,
    definition_ty: Ty<'tcx>,
) -> Result<Ty<'tcx>, ErrorGuaranteed> {
    // Only check this for TAIT. RPIT already supports `tests/ui/impl-trait/nested-return-type2.rs`
    // on stable and we'd break that.
    let opaque_ty_hir = tcx.hir().expect_item(def_id);
    let OpaqueTyOrigin::TyAlias { .. } = opaque_ty_hir.expect_opaque_ty().origin else {
        return Ok(definition_ty);
    };
    let param_env = tcx.param_env(def_id);
    // HACK This bubble is required for this tests to pass:
    // nested-return-type2-tait2.rs
    // nested-return-type2-tait3.rs
    // FIXME(-Ztrait-solver=next): We probably should use `DefiningAnchor::Error`
    // and prepopulate this `InferCtxt` with known opaque values, rather than
    // using the `Bind` anchor here. For now it's fine.
    let infcx = tcx
        .infer_ctxt()
        .with_next_trait_solver(next_trait_solver)
        .with_opaque_type_inference(if next_trait_solver {
            DefiningAnchor::Bind(def_id)
        } else {
            DefiningAnchor::Bubble
        })
        .build();
    let ocx = ObligationCtxt::new(&infcx);
    let identity_substs = InternalSubsts::identity_for_item(tcx, def_id);

    // Require that the hidden type actually fulfills all the bounds of the opaque type, even without
    // the bounds that the function supplies.
    let opaque_ty = tcx.mk_opaque(def_id.to_def_id(), identity_substs);
    ocx.eq(&ObligationCause::misc(definition_span, def_id), param_env, opaque_ty, definition_ty)
        .map_err(|err| {
            infcx
                .err_ctxt()
                .report_mismatched_types(
                    &ObligationCause::misc(definition_span, def_id),
                    opaque_ty,
                    definition_ty,
                    err,
                )
                .emit()
        })?;

    // Require the hidden type to be well-formed with only the generics of the opaque type.
    // Defining use functions may have more bounds than the opaque type, which is ok, as long as the
    // hidden type is well formed even without those bounds.
    let predicate = ty::Binder::dummy(ty::PredicateKind::WellFormed(definition_ty.into()));
    ocx.register_obligation(Obligation::misc(tcx, definition_span, def_id, param_env, predicate));

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.select_all_or_error();

    // This is still required for many(half of the tests in ui/type-alias-impl-trait)
    // tests to pass
    let _ = infcx.take_opaque_types();

    if errors.is_empty() {
        Ok(definition_ty)
    } else {
        Err(infcx.err_ctxt().report_fulfillment_errors(&errors))
    }
}

fn check_opaque_type_parameter_valid(
    tcx: TyCtxt<'_>,
    opaque_type_key: OpaqueTypeKey<'_>,
    span: Span,
) -> Result<(), ErrorGuaranteed> {
    let opaque_ty_hir = tcx.hir().expect_item(opaque_type_key.def_id);
    match opaque_ty_hir.expect_opaque_ty().origin {
        // No need to check return position impl trait (RPIT)
        // because for type and const parameters they are correct
        // by construction: we convert
        //
        // fn foo<P0..Pn>() -> impl Trait
        //
        // into
        //
        // type Foo<P0...Pn>
        // fn foo<P0..Pn>() -> Foo<P0...Pn>.
        //
        // For lifetime parameters we convert
        //
        // fn foo<'l0..'ln>() -> impl Trait<'l0..'lm>
        //
        // into
        //
        // type foo::<'p0..'pn>::Foo<'q0..'qm>
        // fn foo<l0..'ln>() -> foo::<'static..'static>::Foo<'l0..'lm>.
        //
        // which would error here on all of the `'static` args.
        OpaqueTyOrigin::FnReturn(..) | OpaqueTyOrigin::AsyncFn(..) => return Ok(()),
        // Check these
        OpaqueTyOrigin::TyAlias { .. } => {}
    }
    let opaque_generics = tcx.generics_of(opaque_type_key.def_id);
    let mut seen_params: FxIndexMap<_, Vec<_>> = FxIndexMap::default();
    for (i, arg) in opaque_type_key.substs.iter().enumerate() {
        let arg_is_param = match arg.unpack() {
            GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),
            GenericArgKind::Lifetime(lt) => {
                matches!(*lt, ty::ReEarlyBound(_) | ty::ReFree(_))
            }
            GenericArgKind::Const(ct) => matches!(ct.kind(), ty::ConstKind::Param(_)),
        };

        if arg_is_param {
            seen_params.entry(arg).or_default().push(i);
        } else {
            // Prevent `fn foo() -> Foo<u32>` from being defining.
            let opaque_param = opaque_generics.param_at(i, tcx);
            let kind = opaque_param.kind.descr();

            return Err(tcx.sess.emit_err(NonGenericOpaqueTypeParam {
                ty: arg,
                kind,
                span,
                param_span: tcx.def_span(opaque_param.def_id),
            }));
        }
    }

    for (_, indices) in seen_params {
        if indices.len() > 1 {
            let descr = opaque_generics.param_at(indices[0], tcx).kind.descr();
            let spans: Vec<_> = indices
                .into_iter()
                .map(|i| tcx.def_span(opaque_generics.param_at(i, tcx).def_id))
                .collect();
            return Err(tcx
                .sess
                .struct_span_err(span, "non-defining opaque type use in defining scope")
                .span_note(spans, format!("{} used multiple times", descr))
                .emit());
        }
    }

    Ok(())
}
