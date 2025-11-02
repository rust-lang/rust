use rustc_hir::def::DefKind;
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty::{
    self, DefiningScopeKind, DefinitionSiteHiddenType, OpaqueTypeKey, ProvisionalHiddenType,
    TypeVisitableExt, TypingMode,
};
use rustc_trait_selection::error_reporting::infer::need_type_info::TypeAnnotationNeeded;
use rustc_trait_selection::opaque_types::{
    NonDefiningUseReason, opaque_type_has_defining_use_args, report_item_does_not_constrain_error,
};
use rustc_trait_selection::solve;
use tracing::{debug, instrument};

use crate::FnCtxt;

impl<'tcx> FnCtxt<'_, 'tcx> {
    /// This takes all the opaque type uses during HIR typeck. It first computes
    /// the hidden type by iterating over all defining uses.
    ///
    /// A use during HIR typeck is defining if all non-lifetime arguments are
    /// unique generic parameters and the hidden type does not reference any
    /// inference variables.
    ///
    /// It then uses these defining uses to guide inference for all other uses.
    ///
    /// Unlike `handle_opaque_type_uses_next`, this does not report errors.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn try_handle_opaque_type_uses_next(&mut self) {
        // We clone the opaques instead of stealing them here as we still need
        // to use them after fallback.
        let opaque_types: Vec<_> = self.infcx.clone_opaque_types();

        self.compute_definition_site_hidden_types(opaque_types, false);
    }

    /// This takes all the opaque type uses during HIR typeck. It first computes
    /// the concrete hidden type by iterating over all defining uses.
    ///
    /// A use during HIR typeck is defining if all non-lifetime arguments are
    /// unique generic parameters and the hidden type does not reference any
    /// inference variables.
    ///
    /// It then uses these defining uses to guide inference for all other uses.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn handle_opaque_type_uses_next(&mut self) {
        // We clone the opaques instead of stealing them here as they are still used for
        // normalization in the next generation trait solver.
        let opaque_types: Vec<_> = self.infcx.clone_opaque_types();
        let num_entries = self.inner.borrow_mut().opaque_types().num_entries();
        let prev = self.checked_opaque_types_storage_entries.replace(Some(num_entries));
        debug_assert_eq!(prev, None);

        self.compute_definition_site_hidden_types(opaque_types, true);
    }
}

#[derive(Copy, Clone, Debug)]
enum UsageKind<'tcx> {
    None,
    NonDefiningUse(OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>),
    UnconstrainedHiddenType(ProvisionalHiddenType<'tcx>),
    HasDefiningUse(DefinitionSiteHiddenType<'tcx>),
}

impl<'tcx> UsageKind<'tcx> {
    fn merge(&mut self, other: UsageKind<'tcx>) {
        match (&*self, &other) {
            (UsageKind::HasDefiningUse(_), _) | (_, UsageKind::None) => unreachable!(),
            (UsageKind::None, _) => *self = other,
            // When mergining non-defining uses, prefer earlier ones. This means
            // the error happens as early as possible.
            (
                UsageKind::NonDefiningUse(..) | UsageKind::UnconstrainedHiddenType(..),
                UsageKind::NonDefiningUse(..),
            ) => {}
            // When merging unconstrained hidden types, we prefer later ones. This is
            // used as in most cases, the defining use is the final return statement
            // of our function, and other uses with defining arguments are likely not
            // intended to be defining.
            (
                UsageKind::NonDefiningUse(..) | UsageKind::UnconstrainedHiddenType(..),
                UsageKind::UnconstrainedHiddenType(..) | UsageKind::HasDefiningUse(_),
            ) => *self = other,
        }
    }
}

impl<'tcx> FnCtxt<'_, 'tcx> {
    fn compute_definition_site_hidden_types(
        &mut self,
        mut opaque_types: Vec<(OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
        error_on_missing_defining_use: bool,
    ) {
        for entry in opaque_types.iter_mut() {
            *entry = self.resolve_vars_if_possible(*entry);
        }
        debug!(?opaque_types);

        let tcx = self.tcx;
        let TypingMode::Analysis { defining_opaque_types_and_generators } = self.typing_mode()
        else {
            unreachable!();
        };

        for def_id in defining_opaque_types_and_generators {
            match tcx.def_kind(def_id) {
                DefKind::OpaqueTy => {}
                DefKind::Closure => continue,
                _ => unreachable!("not opaque or generator: {def_id:?}"),
            }

            // We do actually need to check this the second pass (we can't just
            // store this), because we can go from `UnconstrainedHiddenType` to
            // `HasDefiningUse` (because of fallback)
            let mut usage_kind = UsageKind::None;
            for &(opaque_type_key, hidden_type) in &opaque_types {
                if opaque_type_key.def_id != def_id {
                    continue;
                }

                usage_kind.merge(self.consider_opaque_type_use(opaque_type_key, hidden_type));

                if let UsageKind::HasDefiningUse(..) = usage_kind {
                    break;
                }
            }

            if let UsageKind::HasDefiningUse(ty) = usage_kind {
                for &(opaque_type_key, hidden_type) in &opaque_types {
                    if opaque_type_key.def_id != def_id {
                        continue;
                    }

                    let expected = ty.ty.instantiate(tcx, opaque_type_key.args);
                    self.demand_eqtype(hidden_type.span, expected, hidden_type.ty);
                }

                // Being explicit here: it may be possible that we in a
                // previous call to this function we did an insert, but this
                // should be just fine, since they all get equated anyways and
                // we shouldn't ever go from `HasDefiningUse` to anyway else.
                let _ = self.typeck_results.borrow_mut().hidden_types.insert(def_id, ty);
            }

            // If we're in `fn try_handle_opaque_type_uses_next` then do not
            // report any errors.
            if !error_on_missing_defining_use {
                continue;
            }

            let guar = match usage_kind {
                UsageKind::HasDefiningUse(_) => continue,
                UsageKind::None => {
                    if let Some(guar) = self.tainted_by_errors() {
                        guar
                    } else {
                        report_item_does_not_constrain_error(self.tcx, self.body_id, def_id, None)
                    }
                }
                UsageKind::NonDefiningUse(opaque_type_key, hidden_type) => {
                    report_item_does_not_constrain_error(
                        self.tcx,
                        self.body_id,
                        def_id,
                        Some((opaque_type_key, hidden_type.span)),
                    )
                }
                UsageKind::UnconstrainedHiddenType(hidden_type) => {
                    if let Some(guar) = self.tainted_by_errors() {
                        guar
                    } else {
                        let infer_var = hidden_type
                            .ty
                            .walk()
                            .filter_map(ty::GenericArg::as_term)
                            .find(|term| term.is_infer())
                            .unwrap_or_else(|| hidden_type.ty.into());
                        self.err_ctxt()
                            .emit_inference_failure_err(
                                self.body_id,
                                hidden_type.span,
                                infer_var,
                                TypeAnnotationNeeded::E0282,
                                false,
                            )
                            .emit()
                    }
                }
            };

            self.typeck_results
                .borrow_mut()
                .hidden_types
                .insert(def_id, DefinitionSiteHiddenType::new_error(tcx, guar));
            self.set_tainted_by_errors(guar);
        }
    }

    #[tracing::instrument(skip(self), ret)]
    fn consider_opaque_type_use(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        hidden_type: ProvisionalHiddenType<'tcx>,
    ) -> UsageKind<'tcx> {
        if let Err(err) = opaque_type_has_defining_use_args(
            &self,
            opaque_type_key,
            hidden_type.span,
            DefiningScopeKind::HirTypeck,
        ) {
            match err {
                NonDefiningUseReason::Tainted(guar) => {
                    return UsageKind::HasDefiningUse(DefinitionSiteHiddenType::new_error(
                        self.tcx, guar,
                    ));
                }
                _ => return UsageKind::NonDefiningUse(opaque_type_key, hidden_type),
            };
        }

        // We ignore uses of the opaque if they have any inference variables
        // as this can frequently happen with recursive calls.
        //
        // See `tests/ui/traits/next-solver/opaques/universal-args-non-defining.rs`.
        if hidden_type.ty.has_non_region_infer() {
            return UsageKind::UnconstrainedHiddenType(hidden_type);
        }

        let cause = ObligationCause::misc(hidden_type.span, self.body_id);
        let at = self.at(&cause, self.param_env);
        let hidden_type = match solve::deeply_normalize(at, hidden_type) {
            Ok(hidden_type) => hidden_type,
            Err(errors) => {
                let guar = self.err_ctxt().report_fulfillment_errors(errors);
                ProvisionalHiddenType::new_error(self.tcx, guar)
            }
        };
        let hidden_type = hidden_type.remap_generic_params_to_declaration_params(
            opaque_type_key,
            self.tcx,
            DefiningScopeKind::HirTypeck,
        );
        UsageKind::HasDefiningUse(hidden_type)
    }

    /// We may in theory add further uses of an opaque after cloning the opaque
    /// types storage during writeback when computing the defining uses.
    ///
    /// Silently ignoring them is dangerous and could result in ICE or even in
    /// unsoundness, so we make sure we catch such cases here. There's currently
    /// no known code where this actually happens, even with the new solver which
    /// does normalize types in writeback after cloning the opaque type storage.
    ///
    /// FIXME(@lcnr): I believe this should be possible in theory and would like
    /// an actual test here. After playing around with this for an hour, I wasn't
    /// able to do anything which didn't already try to normalize the opaque before
    /// then, either allowing compilation to succeed or causing an ambiguity error.
    pub(super) fn detect_opaque_types_added_during_writeback(&self) {
        let num_entries = self.checked_opaque_types_storage_entries.take().unwrap();
        for (key, hidden_type) in
            self.inner.borrow_mut().opaque_types().opaque_types_added_since(num_entries)
        {
            let opaque_type_string = self.tcx.def_path_str(key.def_id);
            let msg = format!("unexpected cyclic definition of `{opaque_type_string}`");
            self.dcx().span_delayed_bug(hidden_type.span, msg);
        }
        let _ = self.take_opaque_types();
    }
}
