// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains code to equate the input/output types appearing
//! in the MIR with the expected input/output types from the function
//! signature. This requires a bit of processing, as the expected types
//! are supplied to us before normalization and may contain existential
//! `impl Trait` instances. In contrast, the input/output types found in
//! the MIR (specifically, in the special local variables for the
//! `RETURN_PLACE` the MIR arguments) are always fully normalize (and
//! contain revealed `impl Trait` values).

use borrow_check::nll::renumber;
use borrow_check::nll::type_check::free_region_relations::UniversalRegionRelations;
use borrow_check::nll::universal_regions::UniversalRegions;
use rustc::hir::def_id::DefId;
use rustc::infer::InferOk;
use rustc::mir::*;
use rustc::traits::query::type_op::custom::CustomTypeOp;
use rustc::traits::{ObligationCause, PredicateObligations};
use rustc::ty::subst::Subst;
use rustc::ty::Ty;

use rustc_data_structures::indexed_vec::Idx;

use super::{Locations, TypeChecker};

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    pub(super) fn equate_inputs_and_outputs(
        &mut self,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
        universal_regions: &UniversalRegions<'tcx>,
        universal_region_relations: &UniversalRegionRelations<'tcx>,
        normalized_inputs_and_output: &[Ty<'tcx>],
    ) {
        let tcx = self.infcx.tcx;

        let (&normalized_output_ty, normalized_input_tys) =
            normalized_inputs_and_output.split_last().unwrap();
        let infcx = self.infcx;

        // Equate expected input tys with those in the MIR.
        let argument_locals = (1..).map(Local::new);
        for (&normalized_input_ty, local) in normalized_input_tys.iter().zip(argument_locals) {
            debug!(
                "equate_inputs_and_outputs: normalized_input_ty = {:?}",
                normalized_input_ty
            );

            let mir_input_ty = mir.local_decls[local].ty;
            self.equate_normalized_input_or_output(normalized_input_ty, mir_input_ty);
        }

        assert!(
            mir.yield_ty.is_some() && universal_regions.yield_ty.is_some()
                || mir.yield_ty.is_none() && universal_regions.yield_ty.is_none()
        );
        if let Some(mir_yield_ty) = mir.yield_ty {
            let ur_yield_ty = universal_regions.yield_ty.unwrap();
            self.equate_normalized_input_or_output(ur_yield_ty, mir_yield_ty);
        }

        // Return types are a bit more complex. They may contain existential `impl Trait`
        // types.
        let param_env = self.param_env;
        let mir_output_ty = mir.local_decls[RETURN_PLACE].ty;
        let opaque_type_map =
            self.fully_perform_op(
                Locations::All,
                CustomTypeOp::new(
                    |infcx| {
                        let mut obligations = ObligationAccumulator::default();

                        let dummy_body_id = ObligationCause::dummy().body_id;
                        let (output_ty, opaque_type_map) =
                            obligations.add(infcx.instantiate_opaque_types(
                                mir_def_id,
                                dummy_body_id,
                                param_env,
                                &normalized_output_ty,
                            ));
                        debug!(
                            "equate_inputs_and_outputs: instantiated output_ty={:?}",
                            output_ty
                        );
                        debug!(
                            "equate_inputs_and_outputs: opaque_type_map={:#?}",
                            opaque_type_map
                        );

                        debug!(
                            "equate_inputs_and_outputs: mir_output_ty={:?}",
                            mir_output_ty
                        );
                        obligations.add(
                            infcx
                                .at(&ObligationCause::dummy(), param_env)
                                .eq(output_ty, mir_output_ty)?,
                        );

                        for (&opaque_def_id, opaque_decl) in &opaque_type_map {
                            let opaque_defn_ty = tcx.type_of(opaque_def_id);
                            let opaque_defn_ty = opaque_defn_ty.subst(tcx, opaque_decl.substs);
                            let opaque_defn_ty = renumber::renumber_regions(
                                infcx,
                                &opaque_defn_ty,
                            );
                            debug!(
                                "equate_inputs_and_outputs: concrete_ty={:?}",
                                opaque_decl.concrete_ty
                            );
                            debug!("equate_inputs_and_outputs: opaque_defn_ty={:?}",
                                   opaque_defn_ty);
                            obligations.add(
                                infcx
                                    .at(&ObligationCause::dummy(), param_env)
                                    .eq(opaque_decl.concrete_ty, opaque_defn_ty)?,
                            );
                        }

                        debug!("equate_inputs_and_outputs: equated");

                        Ok(InferOk {
                            value: Some(opaque_type_map),
                            obligations: obligations.into_vec(),
                        })
                    },
                    || "input_output".to_string(),
                ),
            ).unwrap_or_else(|terr| {
                span_mirbug!(
                    self,
                    Location::START,
                    "equate_inputs_and_outputs: `{:?}=={:?}` failed with `{:?}`",
                    normalized_output_ty,
                    mir_output_ty,
                    terr
                );
                None
            });

        // Finally, if we instantiated the opaque types successfully, we
        // have to solve any bounds (e.g., `-> impl Iterator` needs to
        // prove that `T: Iterator` where `T` is the type we
        // instantiated it with).
        if let Some(opaque_type_map) = opaque_type_map {
            self.fully_perform_op(
                Locations::All,
                CustomTypeOp::new(
                    |_cx| {
                        infcx.constrain_opaque_types(&opaque_type_map, universal_region_relations);
                        Ok(InferOk {
                            value: (),
                            obligations: vec![],
                        })
                    },
                    || "opaque_type_map".to_string(),
                ),
            ).unwrap();
        }
    }

    fn equate_normalized_input_or_output(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) {
        debug!("equate_normalized_input_or_output(a={:?}, b={:?})", a, b);

        if let Err(terr) = self.eq_types(a, b, Locations::All) {
            span_mirbug!(
                self,
                Location::START,
                "equate_normalized_input_or_output: `{:?}=={:?}` failed with `{:?}`",
                a,
                b,
                terr
            );
        }
    }
}

#[derive(Debug, Default)]
struct ObligationAccumulator<'tcx> {
    obligations: PredicateObligations<'tcx>,
}

impl<'tcx> ObligationAccumulator<'tcx> {
    fn add<T>(&mut self, value: InferOk<'tcx, T>) -> T {
        let InferOk { value, obligations } = value;
        self.obligations.extend(obligations);
        value
    }

    fn into_vec(self) -> PredicateObligations<'tcx> {
        self.obligations
    }
}
