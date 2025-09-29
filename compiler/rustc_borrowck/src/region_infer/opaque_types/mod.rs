use std::iter;
use std::rc::Rc;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::outlives::env::RegionBoundPairs;
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin, OpaqueTypeStorageEntries};
use rustc_infer::traits::ObligationCause;
use rustc_macros::extension;
use rustc_middle::mir::{Body, ConstraintCategory, DefinitionSiteHiddenTypes};
use rustc_middle::ty::{
    self, DefiningScopeKind, EarlyBinder, FallibleTypeFolder, GenericArg, GenericArgsRef,
    OpaqueHiddenType, OpaqueTypeKey, Region, RegionVid, Ty, TyCtxt, TypeFoldable,
    TypeSuperFoldable, TypeVisitableExt, fold_regions,
};
use rustc_mir_dataflow::points::DenseLocationMap;
use rustc_span::Span;
use rustc_trait_selection::opaque_types::{
    NonDefiningUseReason, opaque_type_has_defining_use_args,
};
use rustc_trait_selection::solve::NoSolution;
use rustc_trait_selection::traits::query::type_op::custom::CustomTypeOp;
use tracing::{debug, instrument};

use super::reverse_sccs::ReverseSccGraph;
use crate::BorrowckInferCtxt;
use crate::consumers::RegionInferenceContext;
use crate::session_diagnostics::LifetimeMismatchOpaqueParam;
use crate::type_check::canonical::fully_perform_op_raw;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::type_check::{Locations, MirTypeckRegionConstraints};
use crate::universal_regions::{RegionClassification, UniversalRegions};

mod member_constraints;
mod region_ctxt;

use member_constraints::apply_member_constraints;
use region_ctxt::RegionCtxt;

/// We defer errors from [fn handle_opaque_type_uses] and only report them
/// if there are no `RegionErrors`. If there are region errors, it's likely
/// that errors here are caused by them and don't need to be handled separately.
pub(crate) enum DeferredOpaqueTypeError<'tcx> {
    InvalidOpaqueTypeArgs(NonDefiningUseReason<'tcx>),
    LifetimeMismatchOpaqueParam(LifetimeMismatchOpaqueParam<'tcx>),
    UnexpectedHiddenRegion {
        /// The opaque type.
        opaque_type_key: OpaqueTypeKey<'tcx>,
        /// The hidden type containing the member region.
        hidden_type: OpaqueHiddenType<'tcx>,
        /// The unexpected region.
        member_region: Region<'tcx>,
    },
    NonDefiningUseInDefiningScope {
        span: Span,
        opaque_type_key: OpaqueTypeKey<'tcx>,
    },
}

/// We eagerly map all regions to NLL vars here, as we need to make sure we've
/// introduced nll vars for all used placeholders.
///
/// We need to resolve inference vars as even though we're in MIR typeck, we may still
/// encounter inference variables, e.g. when checking user types.
pub(crate) fn clone_and_resolve_opaque_types<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
) -> (OpaqueTypeStorageEntries, Vec<(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)>) {
    let opaque_types = infcx.clone_opaque_types();
    let opaque_types_storage_num_entries = infcx.inner.borrow_mut().opaque_types().num_entries();
    let opaque_types = opaque_types
        .into_iter()
        .map(|entry| {
            fold_regions(infcx.tcx, infcx.resolve_vars_if_possible(entry), |r, _| {
                let vid = if let ty::RePlaceholder(placeholder) = r.kind() {
                    constraints.placeholder_region(infcx, placeholder).as_var()
                } else {
                    universal_region_relations.universal_regions.to_region_vid(r)
                };
                Region::new_var(infcx.tcx, vid)
            })
        })
        .collect::<Vec<_>>();
    (opaque_types_storage_num_entries, opaque_types)
}

/// Maps an NLL var to a deterministically chosen equal universal region.
///
/// See the corresponding [rustc-dev-guide chapter] for more details. This
/// ignores changes to the region values due to member constraints. Applying
/// member constraints does not impact the result of this function.
///
/// [rustc-dev-guide chapter]: https://rustc-dev-guide.rust-lang.org/borrow_check/opaque-types-region-inference-restrictions.html
fn nll_var_to_universal_region<'tcx>(
    rcx: &RegionCtxt<'_, 'tcx>,
    r: RegionVid,
) -> Option<Region<'tcx>> {
    // Use the SCC representative instead of directly using `region`.
    // See [rustc-dev-guide chapter] § "Strict lifetime equality".
    let vid = rcx.representative(r).rvid();
    match rcx.definitions[vid].origin {
        // Iterate over all universal regions in a consistent order and find the
        // *first* equal region. This makes sure that equal lifetimes will have
        // the same name and simplifies subsequent handling.
        // See [rustc-dev-guide chapter] § "Semantic lifetime equality".
        NllRegionVariableOrigin::FreeRegion => rcx
            .universal_regions()
            .universal_regions_iter()
            .filter(|&ur| {
                // See [rustc-dev-guide chapter] § "Closure restrictions".
                !matches!(
                    rcx.universal_regions().region_classification(ur),
                    Some(RegionClassification::External)
                )
            })
            .find(|&ur| rcx.universal_region_relations.equal(vid, ur))
            .map(|ur| rcx.definitions[ur].external_name.unwrap()),
        NllRegionVariableOrigin::Placeholder(placeholder) => {
            Some(ty::Region::new_placeholder(rcx.infcx.tcx, placeholder))
        }
        // If `r` were equal to any universal region, its SCC representative
        // would have been set to a free region.
        NllRegionVariableOrigin::Existential { .. } => None,
    }
}

/// Collect all defining uses of opaque types inside of this typeck root. This
/// expects the hidden type to be mapped to the definition parameters of the opaque
/// and errors if we end up with distinct hidden types.
fn add_hidden_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    hidden_types: &mut DefinitionSiteHiddenTypes<'tcx>,
    def_id: LocalDefId,
    hidden_ty: OpaqueHiddenType<'tcx>,
) {
    // Sometimes two opaque types are the same only after we remap the generic parameters
    // back to the opaque type definition. E.g. we may have `OpaqueType<X, Y>` mapped to
    // `(X, Y)` and `OpaqueType<Y, X>` mapped to `(Y, X)`, and those are the same, but we
    // only know that once we convert the generic parameters to those of the opaque type.
    if let Some(prev) = hidden_types.0.get_mut(&def_id) {
        if prev.ty != hidden_ty.ty {
            let guar = hidden_ty.ty.error_reported().err().unwrap_or_else(|| {
                let (Ok(e) | Err(e)) = prev.build_mismatch_error(&hidden_ty, tcx).map(|d| d.emit());
                e
            });
            prev.ty = Ty::new_error(tcx, guar);
        }
        // Pick a better span if there is one.
        // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
        prev.span = prev.span.substitute_dummy(hidden_ty.span);
    } else {
        hidden_types.0.insert(def_id, hidden_ty);
    }
}

fn get_hidden_type<'tcx>(
    hidden_types: &DefinitionSiteHiddenTypes<'tcx>,
    def_id: LocalDefId,
) -> Option<EarlyBinder<'tcx, OpaqueHiddenType<'tcx>>> {
    hidden_types.0.get(&def_id).map(|ty| EarlyBinder::bind(*ty))
}

#[derive(Debug)]
struct DefiningUse<'tcx> {
    /// The opaque type using non NLL vars. This uses the actual
    /// free regions and placeholders. This is necessary
    /// to interact with code outside of `rustc_borrowck`.
    opaque_type_key: OpaqueTypeKey<'tcx>,
    arg_regions: Vec<RegionVid>,
    hidden_type: OpaqueHiddenType<'tcx>,
}

/// This computes the actual hidden types of the opaque types and maps them to their
/// definition sites. Outside of registering the computed hidden types this function
/// does not mutate the current borrowck state.
///
/// While it may fail to infer the hidden type and return errors, we always apply
/// the computed hidden type to all opaque type uses to check whether they
/// are correct. This is necessary to support non-defining uses of opaques in their
/// defining scope.
///
/// It also means that this whole function is not really soundness critical as we
/// recheck all uses of the opaques regardless.
pub(crate) fn compute_definition_site_hidden_types<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    constraints: &MirTypeckRegionConstraints<'tcx>,
    location_map: Rc<DenseLocationMap>,
    hidden_types: &mut DefinitionSiteHiddenTypes<'tcx>,
    opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
) -> Vec<DeferredOpaqueTypeError<'tcx>> {
    let mut errors = Vec::new();
    // When computing the hidden type we need to track member constraints.
    // We don't mutate the region graph used by `fn compute_regions` but instead
    // manually track region information via a `RegionCtxt`. We discard this
    // information at the end of this function.
    let mut rcx = RegionCtxt::new(infcx, universal_region_relations, location_map, constraints);

    // We start by checking each use of an opaque type during type check and
    // check whether the generic arguments of the opaque type are fully
    // universal, if so, it's a defining use.
    let defining_uses = collect_defining_uses(&mut rcx, hidden_types, opaque_types, &mut errors);

    // We now compute and apply member constraints for all regions in the hidden
    // types of each defining use. This mutates the region values of the `rcx` which
    // is used when mapping the defining uses to the definition site.
    apply_member_constraints(&mut rcx, &defining_uses);

    // After applying member constraints, we now check whether all member regions ended
    // up equal to one of their choice regions and compute the actual hidden type of
    // the opaque type definition. This is stored in the `root_cx`.
    compute_definition_site_hidden_types_from_defining_uses(
        &rcx,
        hidden_types,
        &defining_uses,
        &mut errors,
    );
    errors
}

#[instrument(level = "debug", skip_all, ret)]
fn collect_defining_uses<'tcx>(
    rcx: &mut RegionCtxt<'_, 'tcx>,
    hidden_types: &mut DefinitionSiteHiddenTypes<'tcx>,
    opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
    errors: &mut Vec<DeferredOpaqueTypeError<'tcx>>,
) -> Vec<DefiningUse<'tcx>> {
    let infcx = rcx.infcx;
    let mut defining_uses = vec![];
    for &(opaque_type_key, hidden_type) in opaque_types {
        let non_nll_opaque_type_key = opaque_type_key.fold_captured_lifetime_args(infcx.tcx, |r| {
            nll_var_to_universal_region(&rcx, r.as_var()).unwrap_or(r)
        });
        if let Err(err) = opaque_type_has_defining_use_args(
            infcx,
            non_nll_opaque_type_key,
            hidden_type.span,
            DefiningScopeKind::MirBorrowck,
        ) {
            // A non-defining use. This is a hard error on stable and gets ignored
            // with `TypingMode::Borrowck`.
            if infcx.tcx.use_typing_mode_borrowck() {
                match err {
                    NonDefiningUseReason::Tainted(guar) => add_hidden_type(
                        infcx.tcx,
                        hidden_types,
                        opaque_type_key.def_id,
                        OpaqueHiddenType::new_error(infcx.tcx, guar),
                    ),
                    _ => debug!(?non_nll_opaque_type_key, ?err, "ignoring non-defining use"),
                }
            } else {
                errors.push(DeferredOpaqueTypeError::InvalidOpaqueTypeArgs(err));
            }
            continue;
        }

        // We use the original `opaque_type_key` to compute the `arg_regions`.
        let arg_regions = iter::once(rcx.universal_regions().fr_static)
            .chain(
                opaque_type_key
                    .iter_captured_args(infcx.tcx)
                    .filter_map(|(_, arg)| arg.as_region())
                    .map(Region::as_var),
            )
            .collect();
        defining_uses.push(DefiningUse {
            opaque_type_key: non_nll_opaque_type_key,
            arg_regions,
            hidden_type,
        });
    }

    defining_uses
}

fn compute_definition_site_hidden_types_from_defining_uses<'tcx>(
    rcx: &RegionCtxt<'_, 'tcx>,
    hidden_types: &mut DefinitionSiteHiddenTypes<'tcx>,
    defining_uses: &[DefiningUse<'tcx>],
    errors: &mut Vec<DeferredOpaqueTypeError<'tcx>>,
) {
    let infcx = rcx.infcx;
    let tcx = infcx.tcx;
    let mut decls_modulo_regions: FxIndexMap<OpaqueTypeKey<'tcx>, (OpaqueTypeKey<'tcx>, Span)> =
        FxIndexMap::default();
    for &DefiningUse { opaque_type_key, ref arg_regions, hidden_type } in defining_uses {
        // After applying member constraints, we now map all regions in the hidden type
        // to the `arg_regions` of this defining use. In case a region in the hidden type
        // ended up not being equal to any such region, we error.
        let hidden_type =
            match hidden_type.try_fold_with(&mut ToArgRegionsFolder::new(rcx, arg_regions)) {
                Ok(hidden_type) => hidden_type,
                Err(r) => {
                    errors.push(DeferredOpaqueTypeError::UnexpectedHiddenRegion {
                        hidden_type,
                        opaque_type_key,
                        member_region: ty::Region::new_var(tcx, r),
                    });
                    let guar = tcx.dcx().span_delayed_bug(
                        hidden_type.span,
                        "opaque type with non-universal region args",
                    );
                    ty::OpaqueHiddenType::new_error(tcx, guar)
                }
            };

        // Now that we mapped the member regions to their final value,
        // map the arguments of the opaque type key back to the parameters
        // of the opaque type definition.
        let ty = infcx
            .infer_opaque_definition_from_instantiation(opaque_type_key, hidden_type)
            .unwrap_or_else(|_| {
                Ty::new_error_with_message(
                    rcx.infcx.tcx,
                    hidden_type.span,
                    "deferred invalid opaque type args",
                )
            });

        // Sometimes, when the hidden type is an inference variable, it can happen that
        // the hidden type becomes the opaque type itself. In this case, this was an opaque
        // usage of the opaque type and we can ignore it. This check is mirrored in typeck's
        // writeback.
        if !rcx.infcx.tcx.use_typing_mode_borrowck() {
            if let ty::Alias(ty::Opaque, alias_ty) = ty.kind()
                && alias_ty.def_id == opaque_type_key.def_id.to_def_id()
                && alias_ty.args == opaque_type_key.args
            {
                continue;
            }
        }

        // Check that all opaque types have the same region parameters if they have the same
        // non-region parameters. This is necessary because within the new solver we perform
        // various query operations modulo regions, and thus could unsoundly select some impls
        // that don't hold.
        //
        // FIXME(-Znext-solver): This isn't necessary after all. We can remove this check again.
        if let Some((prev_decl_key, prev_span)) = decls_modulo_regions.insert(
            rcx.infcx.tcx.erase_and_anonymize_regions(opaque_type_key),
            (opaque_type_key, hidden_type.span),
        ) && let Some((arg1, arg2)) = std::iter::zip(
            prev_decl_key.iter_captured_args(infcx.tcx).map(|(_, arg)| arg),
            opaque_type_key.iter_captured_args(infcx.tcx).map(|(_, arg)| arg),
        )
        .find(|(arg1, arg2)| arg1 != arg2)
        {
            errors.push(DeferredOpaqueTypeError::LifetimeMismatchOpaqueParam(
                LifetimeMismatchOpaqueParam {
                    arg: arg1,
                    prev: arg2,
                    span: prev_span,
                    prev_span: hidden_type.span,
                },
            ));
        }
        add_hidden_type(
            tcx,
            hidden_types,
            opaque_type_key.def_id,
            OpaqueHiddenType { span: hidden_type.span, ty },
        );
    }
}

/// A folder to map the regions in the hidden type to their corresponding `arg_regions`.
///
/// This folder has to differentiate between member regions and other regions in the hidden
/// type. Member regions have to be equal to one of the `arg_regions` while other regions simply
/// get treated as an existential region in the opaque if they are not. Existential
/// regions are currently represented using `'erased`.
struct ToArgRegionsFolder<'a, 'tcx> {
    rcx: &'a RegionCtxt<'a, 'tcx>,
    // When folding closure args or bivariant alias arguments, we simply
    // ignore non-member regions. However, we still need to map member
    // regions to their arg region even if its in a closure argument.
    //
    // See tests/ui/type-alias-impl-trait/closure_wf_outlives.rs for an example.
    erase_unknown_regions: bool,
    arg_regions: &'a [RegionVid],
}

impl<'a, 'tcx> ToArgRegionsFolder<'a, 'tcx> {
    fn new(
        rcx: &'a RegionCtxt<'a, 'tcx>,
        arg_regions: &'a [RegionVid],
    ) -> ToArgRegionsFolder<'a, 'tcx> {
        ToArgRegionsFolder { rcx, erase_unknown_regions: false, arg_regions }
    }

    fn fold_non_member_arg(&mut self, arg: GenericArg<'tcx>) -> GenericArg<'tcx> {
        let prev = self.erase_unknown_regions;
        self.erase_unknown_regions = true;
        let res = arg.try_fold_with(self).unwrap();
        self.erase_unknown_regions = prev;
        res
    }

    fn fold_closure_args(
        &mut self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Result<GenericArgsRef<'tcx>, RegionVid> {
        let generics = self.cx().generics_of(def_id);
        self.cx().mk_args_from_iter(args.iter().enumerate().map(|(index, arg)| {
            if index < generics.parent_count {
                Ok(self.fold_non_member_arg(arg))
            } else {
                arg.try_fold_with(self)
            }
        }))
    }
}
impl<'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for ToArgRegionsFolder<'_, 'tcx> {
    type Error = RegionVid;
    fn cx(&self) -> TyCtxt<'tcx> {
        self.rcx.infcx.tcx
    }

    fn try_fold_region(&mut self, r: Region<'tcx>) -> Result<Region<'tcx>, RegionVid> {
        match r.kind() {
            // ignore bound regions, keep visiting
            ty::ReBound(_, _) => Ok(r),
            _ => {
                let r = r.as_var();
                if let Some(arg_region) = self
                    .arg_regions
                    .iter()
                    .copied()
                    .find(|&arg_vid| self.rcx.eval_equal(r, arg_vid))
                    .and_then(|r| nll_var_to_universal_region(self.rcx, r))
                {
                    Ok(arg_region)
                } else if self.erase_unknown_regions {
                    Ok(self.cx().lifetimes.re_erased)
                } else {
                    Err(r)
                }
            }
        }
    }

    fn try_fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, RegionVid> {
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return Ok(ty);
        }

        let tcx = self.cx();
        Ok(match *ty.kind() {
            ty::Closure(def_id, args) => {
                Ty::new_closure(tcx, def_id, self.fold_closure_args(def_id, args)?)
            }

            ty::CoroutineClosure(def_id, args) => {
                Ty::new_coroutine_closure(tcx, def_id, self.fold_closure_args(def_id, args)?)
            }

            ty::Coroutine(def_id, args) => {
                Ty::new_coroutine(tcx, def_id, self.fold_closure_args(def_id, args)?)
            }

            ty::Alias(kind, ty::AliasTy { def_id, args, .. })
                if let Some(variances) = tcx.opt_alias_variances(kind, def_id) =>
            {
                let args = tcx.mk_args_from_iter(std::iter::zip(variances, args.iter()).map(
                    |(&v, s)| {
                        if v == ty::Bivariant {
                            Ok(self.fold_non_member_arg(s))
                        } else {
                            s.try_fold_with(self)
                        }
                    },
                ))?;
                ty::AliasTy::new_from_args(tcx, def_id, args).to_ty(tcx)
            }

            _ => ty.try_super_fold_with(self)?,
        })
    }
}

/// This function is what actually applies member constraints to the borrowck
/// state. It is also responsible to check all uses of the opaques in their
/// defining scope.
///
/// It does this by equating the hidden type of each use with the instantiated final
/// hidden type of the opaque.
pub(crate) fn apply_definition_site_hidden_types<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    universal_regions: &UniversalRegions<'tcx>,
    region_bound_pairs: &RegionBoundPairs<'tcx>,
    known_type_outlives_obligations: &[ty::PolyTypeOutlivesPredicate<'tcx>],
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
    hidden_types: &mut DefinitionSiteHiddenTypes<'tcx>,
    opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
) -> Vec<DeferredOpaqueTypeError<'tcx>> {
    let tcx = infcx.tcx;
    let mut errors = Vec::new();
    for &(key, hidden_type) in opaque_types {
        let Some(expected) = get_hidden_type(hidden_types, key.def_id) else {
            if !tcx.use_typing_mode_borrowck() {
                if let ty::Alias(ty::Opaque, alias_ty) = hidden_type.ty.kind()
                    && alias_ty.def_id == key.def_id.to_def_id()
                    && alias_ty.args == key.args
                {
                    continue;
                } else {
                    unreachable!("non-defining use in defining scope");
                }
            }
            errors.push(DeferredOpaqueTypeError::NonDefiningUseInDefiningScope {
                span: hidden_type.span,
                opaque_type_key: key,
            });
            let guar = tcx.dcx().span_delayed_bug(
                hidden_type.span,
                "non-defining use in the defining scope with no defining uses",
            );
            add_hidden_type(tcx, hidden_types, key.def_id, OpaqueHiddenType::new_error(tcx, guar));
            continue;
        };

        // We erase all non-member region of the opaque and need to treat these as existentials.
        let expected = ty::fold_regions(tcx, expected.instantiate(tcx, key.args), |re, _dbi| {
            match re.kind() {
                ty::ReErased => infcx.next_nll_region_var(
                    NllRegionVariableOrigin::Existential { name: None },
                    || crate::RegionCtxt::Existential(None),
                ),
                _ => re,
            }
        });

        // We now simply equate the expected with the actual hidden type.
        let locations = Locations::All(hidden_type.span);
        if let Err(guar) = fully_perform_op_raw(
            infcx,
            body,
            universal_regions,
            region_bound_pairs,
            known_type_outlives_obligations,
            constraints,
            locations,
            ConstraintCategory::OpaqueType,
            CustomTypeOp::new(
                |ocx| {
                    let cause = ObligationCause::misc(
                        hidden_type.span,
                        body.source.def_id().expect_local(),
                    );
                    // We need to normalize both types in the old solver before equatingt them.
                    let actual_ty = ocx.normalize(&cause, infcx.param_env, hidden_type.ty);
                    let expected_ty = ocx.normalize(&cause, infcx.param_env, expected.ty);
                    ocx.eq(&cause, infcx.param_env, actual_ty, expected_ty).map_err(|_| NoSolution)
                },
                "equating opaque types",
            ),
        ) {
            add_hidden_type(tcx, hidden_types, key.def_id, OpaqueHiddenType::new_error(tcx, guar));
        }
    }
    errors
}

/// In theory `apply_definition_site_hidden_types` could introduce new uses of opaque types.
/// We do not check these new uses so this could be unsound.
///
/// We detect any new uses and simply delay a bug if they occur. If this results in
/// an ICE we can properly handle this, but we haven't encountered any such test yet.
///
/// See the related comment in `FnCtxt::detect_opaque_types_added_during_writeback`.
pub(crate) fn detect_opaque_types_added_while_handling_opaque_types<'tcx>(
    infcx: &InferCtxt<'tcx>,
    opaque_types_storage_num_entries: OpaqueTypeStorageEntries,
) {
    for (key, hidden_type) in infcx
        .inner
        .borrow_mut()
        .opaque_types()
        .opaque_types_added_since(opaque_types_storage_num_entries)
    {
        let opaque_type_string = infcx.tcx.def_path_str(key.def_id);
        let msg = format!("unexpected cyclic definition of `{opaque_type_string}`");
        infcx.dcx().span_delayed_bug(hidden_type.span, msg);
    }

    let _ = infcx.take_opaque_types();
}

impl<'tcx> RegionInferenceContext<'tcx> {
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
        fold_regions(tcx, ty, |region, _| match region.kind() {
            ty::ReVar(vid) => {
                let scc = self.constraint_sccs.scc(vid);

                // Special handling of higher-ranked regions.
                if !self.max_nameable_universe(scc).is_root() {
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
                let rev_scc_graph =
                    ReverseSccGraph::compute(&self.constraint_sccs, self.universal_regions());
                let upper_bounds: Vec<_> = rev_scc_graph
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
    #[instrument(level = "debug", skip(self))]
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
    ) -> Result<Ty<'tcx>, NonDefiningUseReason<'tcx>> {
        opaque_type_has_defining_use_args(
            self,
            opaque_type_key,
            instantiated_ty.span,
            DefiningScopeKind::MirBorrowck,
        )?;

        let definition_ty = instantiated_ty
            .remap_generic_params_to_declaration_params(
                opaque_type_key,
                self.tcx,
                DefiningScopeKind::MirBorrowck,
            )
            .ty;

        definition_ty.error_reported()?;
        Ok(definition_ty)
    }
}
