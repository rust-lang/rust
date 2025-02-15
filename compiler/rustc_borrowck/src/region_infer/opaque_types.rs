use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::OpaqueTyOrigin;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin, TyCtxtInferExt as _};
use rustc_macros::extension;
use rustc_middle::ty::fold::fold_regions;
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{
    self, GenericArgKind, GenericArgs, OpaqueHiddenType, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable,
    TypingMode,
};
use rustc_span::Span;
use rustc_trait_selection::regions::OutlivesEnvironmentBuildExt;
use rustc_trait_selection::traits::ObligationCtxt;
use tracing::{debug, instrument};

use super::RegionInferenceContext;
use crate::session_diagnostics::{LifetimeMismatchOpaqueParam, NonGenericOpaqueTypeParam};
use crate::universal_regions::RegionClassification;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Resolve any opaque types that were encountered while borrow checking
    /// this item. This is then used to get the type in the `type_of` query.
    ///
    /// For example consider `fn f<'a>(x: &'a i32) -> impl Sized + 'a { x }`.
    /// This is lowered to give HIR something like
    ///
    /// type f<'a>::_Return<'_x> = impl Sized + '_x;
    /// fn f<'a>(x: &'a i32) -> f<'a>::_Return<'a> { x }
    ///
    /// When checking the return type record the type from the return and the
    /// type used in the return value. In this case they might be `_Return<'1>`
    /// and `&'2 i32` respectively.
    ///
    /// Once we to this method, we have completed region inference and want to
    /// call `infer_opaque_definition_from_instantiation` to get the inferred
    /// type of `_Return<'_x>`. `infer_opaque_definition_from_instantiation`
    /// compares lifetimes directly, so we need to map the inference variables
    /// back to concrete lifetimes: `'static`, `ReEarlyParam` or `ReLateParam`.
    ///
    /// First we map the regions in the generic parameters `_Return<'1>` to
    /// their `external_name` giving `_Return<'a>`. This step is a bit involved.
    /// See the [rustc-dev-guide chapter] for more info.
    ///
    /// Then we map all the lifetimes in the concrete type to an equal
    /// universal region that occurs in the opaque type's args, in this case
    /// this would result in `&'a i32`. We only consider regions in the args
    /// in case there is an equal region that does not. For example, this should
    /// be allowed:
    /// `fn f<'a: 'b, 'b: 'a>(x: *mut &'b i32) -> impl Sized + 'a { x }`
    ///
    /// This will then allow `infer_opaque_definition_from_instantiation` to
    /// determine that `_Return<'_x> = &'_x i32`.
    ///
    /// There's a slight complication around closures. Given
    /// `fn f<'a: 'a>() { || {} }` the closure's type is something like
    /// `f::<'a>::{{closure}}`. The region parameter from f is essentially
    /// ignored by type checking so ends up being inferred to an empty region.
    /// Calling `universal_upper_bound` for such a region gives `fr_fn_body`,
    /// which has no `external_name` in which case we use `'{erased}` as the
    /// region to pass to `infer_opaque_definition_from_instantiation`.
    ///
    /// [rustc-dev-guide chapter]:
    /// https://rustc-dev-guide.rust-lang.org/opaque-types-region-infer-restrictions.html
    #[instrument(level = "debug", skip(self, infcx), ret)]
    pub(crate) fn infer_opaque_types(
        &self,
        infcx: &InferCtxt<'tcx>,
        opaque_ty_decls: FxIndexMap<OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>>,
    ) -> FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>> {
        let mut result: FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>> = FxIndexMap::default();
        let mut decls_modulo_regions: FxIndexMap<OpaqueTypeKey<'tcx>, (OpaqueTypeKey<'tcx>, Span)> =
            FxIndexMap::default();

        for (opaque_type_key, concrete_type) in opaque_ty_decls {
            debug!(?opaque_type_key, ?concrete_type);

            let mut arg_regions: Vec<(ty::RegionVid, ty::Region<'_>)> =
                vec![(self.universal_regions().fr_static, infcx.tcx.lifetimes.re_static)];

            let opaque_type_key =
                opaque_type_key.fold_captured_lifetime_args(infcx.tcx, |region| {
                    // Use the SCC representative instead of directly using `region`.
                    // See [rustc-dev-guide chapter] § "Strict lifetime equality".
                    let scc = self.constraint_sccs.scc(region.as_var());
                    let vid = self.scc_representative(scc);
                    let named = match self.definitions[vid].origin {
                        // Iterate over all universal regions in a consistent order and find the
                        // *first* equal region. This makes sure that equal lifetimes will have
                        // the same name and simplifies subsequent handling.
                        // See [rustc-dev-guide chapter] § "Semantic lifetime equality".
                        NllRegionVariableOrigin::FreeRegion => self
                            .universal_regions()
                            .universal_regions_iter()
                            .filter(|&ur| {
                                // See [rustc-dev-guide chapter] § "Closure restrictions".
                                !matches!(
                                    self.universal_regions().region_classification(ur),
                                    Some(RegionClassification::External)
                                )
                            })
                            .find(|&ur| self.universal_region_relations.equal(vid, ur))
                            .map(|ur| self.definitions[ur].external_name.unwrap()),
                        NllRegionVariableOrigin::Placeholder(placeholder) => {
                            Some(ty::Region::new_placeholder(infcx.tcx, placeholder))
                        }
                        NllRegionVariableOrigin::Existential { .. } => None,
                    }
                    .unwrap_or_else(|| {
                        ty::Region::new_error_with_message(
                            infcx.tcx,
                            concrete_type.span,
                            "opaque type with non-universal region args",
                        )
                    });

                    arg_regions.push((vid, named));
                    named
                });
            debug!(?opaque_type_key, ?arg_regions);

            let concrete_type = fold_regions(infcx.tcx, concrete_type, |region, _| {
                arg_regions
                    .iter()
                    .find(|&&(arg_vid, _)| self.eval_equal(region.as_var(), arg_vid))
                    .map(|&(_, arg_named)| arg_named)
                    .unwrap_or(infcx.tcx.lifetimes.re_erased)
            });
            debug!(?concrete_type);

            let ty =
                infcx.infer_opaque_definition_from_instantiation(opaque_type_key, concrete_type);

            // Sometimes, when the hidden type is an inference variable, it can happen that
            // the hidden type becomes the opaque type itself. In this case, this was an opaque
            // usage of the opaque type and we can ignore it. This check is mirrored in typeck's
            // writeback.
            if !infcx.next_trait_solver() {
                if let ty::Alias(ty::Opaque, alias_ty) = ty.kind()
                    && alias_ty.def_id == opaque_type_key.def_id.to_def_id()
                    && alias_ty.args == opaque_type_key.args
                {
                    continue;
                }
            }
            // Sometimes two opaque types are the same only after we remap the generic parameters
            // back to the opaque type definition. E.g. we may have `OpaqueType<X, Y>` mapped to
            // `(X, Y)` and `OpaqueType<Y, X>` mapped to `(Y, X)`, and those are the same, but we
            // only know that once we convert the generic parameters to those of the opaque type.
            if let Some(prev) = result.get_mut(&opaque_type_key.def_id) {
                if prev.ty != ty {
                    let guar = ty.error_reported().err().unwrap_or_else(|| {
                        let (Ok(e) | Err(e)) = prev
                            .build_mismatch_error(
                                &OpaqueHiddenType { ty, span: concrete_type.span },
                                infcx.tcx,
                            )
                            .map(|d| d.emit());
                        e
                    });
                    prev.ty = Ty::new_error(infcx.tcx, guar);
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

            // Check that all opaque types have the same region parameters if they have the same
            // non-region parameters. This is necessary because within the new solver we perform
            // various query operations modulo regions, and thus could unsoundly select some impls
            // that don't hold.
            if !ty.references_error()
                && let Some((prev_decl_key, prev_span)) = decls_modulo_regions.insert(
                    infcx.tcx.erase_regions(opaque_type_key),
                    (opaque_type_key, concrete_type.span),
                )
                && let Some((arg1, arg2)) = std::iter::zip(
                    prev_decl_key.iter_captured_args(infcx.tcx).map(|(_, arg)| arg),
                    opaque_type_key.iter_captured_args(infcx.tcx).map(|(_, arg)| arg),
                )
                .find(|(arg1, arg2)| arg1 != arg2)
            {
                infcx.dcx().emit_err(LifetimeMismatchOpaqueParam {
                    arg: arg1,
                    prev: arg2,
                    span: prev_span,
                    prev_span: concrete_type.span,
                });
            }
        }
        result
    }

    /// Map the regions in the type to named regions. This is similar to what
    /// `infer_opaque_types` does, but can infer any universal region, not only
    /// ones from the args for the opaque type. It also doesn't double check
    /// that the regions produced are in fact equal to the named region they are
    /// replaced with. This is fine because this function is only to improve the
    /// region names in error messages.
    ///
    /// This differs from `MirBorrowckCtxt::name_regions` since it is particularly
    /// lax with mapping region vids that are *shorter* than a universal region to
    /// that universal region. This is useful for member region constraints since
    /// we want to suggest a universal region name to capture even if it's technically
    /// not equal to the error region.
    pub(crate) fn name_regions_for_member_constraint<T>(&self, tcx: TyCtxt<'tcx>, ty: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        fold_regions(tcx, ty, |region, _| match *region {
            ty::ReVar(vid) => {
                let scc = self.constraint_sccs.scc(vid);

                // Special handling of higher-ranked regions.
                if !self.scc_universe(scc).is_root() {
                    match self.scc_values.placeholders_contained_in(scc).enumerate().last() {
                        // If the region contains a single placeholder then they're equal.
                        Some((0, placeholder)) => {
                            return ty::Region::new_placeholder(tcx, placeholder);
                        }

                        // Fallback: this will produce a cryptic error message.
                        _ => return region,
                    }
                }

                // Find something that we can name
                let upper_bound = self.approx_universal_upper_bound(vid);
                if let Some(universal_region) = self.definitions[upper_bound].external_name {
                    return universal_region;
                }

                // Nothing exact found, so we pick a named upper bound, if there's only one.
                // If there's >1 universal region, then we probably are dealing w/ an intersection
                // region which cannot be mapped back to a universal.
                // FIXME: We could probably compute the LUB if there is one.
                let scc = self.constraint_sccs.scc(vid);
                let upper_bounds: Vec<_> = self
                    .rev_scc_graph
                    .as_ref()
                    .unwrap()
                    .upper_bounds(scc)
                    .filter_map(|vid| self.definitions[vid].external_name)
                    .filter(|r| !r.is_static())
                    .collect();
                match &upper_bounds[..] {
                    [universal_region] => *universal_region,
                    _ => region,
                }
            }
            _ => region,
        })
    }
}

#[extension(pub trait InferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
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
    /// - `args`, the args used to instantiate this opaque type
    /// - `instantiated_ty`, the inferred type C1 -- fully resolved, lifted version of
    ///   `opaque_defn.concrete_ty`
    #[instrument(level = "debug", skip(self))]
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
    ) -> Ty<'tcx> {
        if let Some(e) = self.tainted_by_errors() {
            return Ty::new_error(self.tcx, e);
        }

        if let Err(guar) =
            check_opaque_type_parameter_valid(self, opaque_type_key, instantiated_ty.span)
        {
            return Ty::new_error(self.tcx, guar);
        }

        let definition_ty = instantiated_ty
            .remap_generic_params_to_declaration_params(opaque_type_key, self.tcx, false)
            .ty;

        if let Err(e) = definition_ty.error_reported() {
            return Ty::new_error(self.tcx, e);
        }

        definition_ty
    }
}

/// Opaque type parameter validity check as documented in the [rustc-dev-guide chapter].
///
/// [rustc-dev-guide chapter]:
/// https://rustc-dev-guide.rust-lang.org/opaque-types-region-infer-restrictions.html
fn check_opaque_type_parameter_valid<'tcx>(
    infcx: &InferCtxt<'tcx>,
    opaque_type_key: OpaqueTypeKey<'tcx>,
    span: Span,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    let opaque_generics = tcx.generics_of(opaque_type_key.def_id);
    let opaque_env = LazyOpaqueTyEnv::new(tcx, opaque_type_key.def_id);
    let mut seen_params: FxIndexMap<_, Vec<_>> = FxIndexMap::default();

    for (i, arg) in opaque_type_key.iter_captured_args(tcx) {
        let arg_is_param = match arg.unpack() {
            GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),
            GenericArgKind::Lifetime(lt) => {
                matches!(*lt, ty::ReEarlyParam(_) | ty::ReLateParam(_))
                    || (lt.is_static() && opaque_env.param_equal_static(i))
            }
            GenericArgKind::Const(ct) => matches!(ct.kind(), ty::ConstKind::Param(_)),
        };

        if arg_is_param {
            // Register if the same lifetime appears multiple times in the generic args.
            // There is an exception when the opaque type *requires* the lifetimes to be equal.
            // See [rustc-dev-guide chapter] § "An exception to uniqueness rule".
            let seen_where = seen_params.entry(arg).or_default();
            if !seen_where.first().is_some_and(|&prev_i| opaque_env.params_equal(i, prev_i)) {
                seen_where.push(i);
            }
        } else {
            // Prevent `fn foo() -> Foo<u32>` from being defining.
            let opaque_param = opaque_generics.param_at(i, tcx);
            let kind = opaque_param.kind.descr();

            opaque_env.param_is_error(i)?;

            return Err(infcx.dcx().emit_err(NonGenericOpaqueTypeParam {
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
            #[allow(rustc::diagnostic_outside_of_impl)]
            #[allow(rustc::untranslatable_diagnostic)]
            return Err(infcx
                .dcx()
                .struct_span_err(span, "non-defining opaque type use in defining scope")
                .with_span_note(spans, format!("{descr} used multiple times"))
                .emit());
        }
    }

    Ok(())
}

/// Computes if an opaque type requires a lifetime parameter to be equal to
/// another one or to the `'static` lifetime.
/// These requirements are derived from the explicit and implied bounds.
struct LazyOpaqueTyEnv<'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,

    /// Equal parameters will have the same name. Computed Lazily.
    /// Example:
    ///     `type Opaque<'a: 'static, 'b: 'c, 'c: 'b> = impl Sized;`
    ///     Identity args: `['a, 'b, 'c]`
    ///     Canonical args: `['static, 'b, 'b]`
    canonical_args: std::cell::OnceCell<ty::GenericArgsRef<'tcx>>,
}

impl<'tcx> LazyOpaqueTyEnv<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        Self { tcx, def_id, canonical_args: std::cell::OnceCell::new() }
    }

    fn param_equal_static(&self, param_index: usize) -> bool {
        self.get_canonical_args()[param_index].expect_region().is_static()
    }

    fn params_equal(&self, param1: usize, param2: usize) -> bool {
        let canonical_args = self.get_canonical_args();
        canonical_args[param1] == canonical_args[param2]
    }

    fn param_is_error(&self, param_index: usize) -> Result<(), ErrorGuaranteed> {
        self.get_canonical_args()[param_index].error_reported()
    }

    fn get_canonical_args(&self) -> ty::GenericArgsRef<'tcx> {
        if let Some(&canonical_args) = self.canonical_args.get() {
            return canonical_args;
        }

        let &Self { tcx, def_id, .. } = self;
        let origin = tcx.local_opaque_ty_origin(def_id);
        let parent = match origin {
            OpaqueTyOrigin::FnReturn { parent, .. }
            | OpaqueTyOrigin::AsyncFn { parent, .. }
            | OpaqueTyOrigin::TyAlias { parent, .. } => parent,
        };
        let param_env = tcx.param_env(parent);
        let args = GenericArgs::identity_for_item(tcx, parent).extend_to(
            tcx,
            def_id.to_def_id(),
            |param, _| {
                tcx.map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local()).into()
            },
        );

        // FIXME(#132279): It feels wrong to use `non_body_analysis` here given that we're
        // in a body here.
        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let ocx = ObligationCtxt::new(&infcx);

        let wf_tys = ocx.assumed_wf_types(param_env, parent).unwrap_or_else(|_| {
            tcx.dcx().span_delayed_bug(tcx.def_span(def_id), "error getting implied bounds");
            Default::default()
        });
        let outlives_env = OutlivesEnvironment::new(&infcx, parent, param_env, wf_tys);

        let mut seen = vec![tcx.lifetimes.re_static];
        let canonical_args = fold_regions(tcx, args, |r1, _| {
            if r1.is_error() {
                r1
            } else if let Some(&r2) = seen.iter().find(|&&r2| {
                let free_regions = outlives_env.free_region_map();
                free_regions.sub_free_regions(tcx, r1, r2)
                    && free_regions.sub_free_regions(tcx, r2, r1)
            }) {
                r2
            } else {
                seen.push(r1);
                r1
            }
        });
        self.canonical_args.set(canonical_args).unwrap();
        canonical_args
    }
}
