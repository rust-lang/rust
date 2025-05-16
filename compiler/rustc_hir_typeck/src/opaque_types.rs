use rustc_middle::ty::{
    DefiningScopeKind, EarlyBinder, OpaqueHiddenType, OpaqueTypeKey, Ty, TypeVisitableExt,
};
use rustc_trait_selection::opaque_types::{
    InvalidOpaqueTypeArgs, check_opaque_type_parameter_valid,
};
use tracing::{debug, instrument};

use crate::FnCtxt;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    pub(super) fn handle_opaque_type_uses_next(&mut self) {
        // We clone the opaques instead of stealing them here as they are still used for
        // normalization in the next generation trait solver.
        //
        // FIXME(-Znext-solver): Opaque types defined after this would simply get dropped
        // at the end of typeck. Ideally we can feed some query here to no longer define
        // new opaque uses but instead always reveal by using the definitions inferred here.
        let mut opaque_types: Vec<_> = self.infcx.clone_opaque_types();
        let num_entries = self.inner.borrow_mut().opaque_types().num_entries();
        let prev = self.checked_opaque_types_storage_entries.replace(Some(num_entries));
        debug_assert_eq!(prev, None);
        for entry in &mut opaque_types {
            *entry = self.resolve_vars_if_possible(*entry);
        }
        debug!(?opaque_types);

        self.collect_defining_uses(&opaque_types);
        self.apply_defining_uses(&opaque_types);
    }

    fn collect_defining_uses(
        &mut self,
        opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
    ) {
        let tcx = self.tcx;
        let typeck_results = &mut *self.typeck_results.borrow_mut();
        for &(opaque_type_key, hidden_type) in opaque_types {
            match check_opaque_type_parameter_valid(
                &self,
                opaque_type_key,
                hidden_type.span,
                DefiningScopeKind::HirTypeck,
            ) {
                Ok(()) => {}
                Err(InvalidOpaqueTypeArgs::AlreadyReported(guar)) => {
                    typeck_results
                        .concrete_opaque_types
                        .insert(opaque_type_key.def_id, OpaqueHiddenType::new_error(tcx, guar));
                }
                // Not a defining use, ignore and treat as revealing use instead.
                Err(
                    InvalidOpaqueTypeArgs::NotAParam { .. }
                    | InvalidOpaqueTypeArgs::DuplicateParam { .. },
                ) => continue,
            }

            // We ignore uses of the opaque if they have any inference variables
            // as this can frequently happen with recursive calls.
            //
            // See `tests/ui/traits/next-solver/opaques/universal-args-non-defining.rs`.
            if hidden_type.ty.has_non_region_infer() {
                continue;
            }

            let hidden_type = hidden_type.remap_generic_params_to_declaration_params(
                opaque_type_key,
                tcx,
                DefiningScopeKind::HirTypeck,
            );

            if let Some(prev) =
                typeck_results.concrete_opaque_types.insert(opaque_type_key.def_id, hidden_type)
            {
                let entry =
                    typeck_results.concrete_opaque_types.get_mut(&opaque_type_key.def_id).unwrap();
                if prev.ty != hidden_type.ty {
                    if let Some(guar) = typeck_results.tainted_by_errors {
                        entry.ty = Ty::new_error(tcx, guar);
                    } else {
                        let (Ok(guar) | Err(guar)) =
                            prev.build_mismatch_error(&hidden_type, tcx).map(|d| d.emit());
                        entry.ty = Ty::new_error(tcx, guar);
                    }
                }

                // Pick a better span if there is one.
                // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
                entry.span = prev.span.substitute_dummy(hidden_type.span);
            }
        }

        // FIXME(-Znext-solver): Check that all opaques have been defined hre.
    }

    fn apply_defining_uses(
        &mut self,
        opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
    ) {
        let tcx = self.tcx;
        for &(key, hidden_type) in opaque_types {
            let Some(&expected) =
                self.typeck_results.borrow_mut().concrete_opaque_types.get(&key.def_id)
            else {
                let guar =
                    tcx.dcx().span_err(hidden_type.span, "non-defining use in the defining scope");
                self.typeck_results
                    .borrow_mut()
                    .concrete_opaque_types
                    .insert(key.def_id, OpaqueHiddenType::new_error(tcx, guar));
                self.set_tainted_by_errors(guar);
                continue;
            };

            let expected = EarlyBinder::bind(expected.ty).instantiate(tcx, key.args);
            self.demand_eqtype(hidden_type.span, expected, hidden_type.ty);
        }
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
