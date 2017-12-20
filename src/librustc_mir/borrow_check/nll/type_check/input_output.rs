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
use borrow_check::nll::universal_regions::UniversalRegions;
use rustc::hir::def_id::DefId;
use rustc::infer::InferOk;
use rustc::ty::Ty;
use rustc::ty::subst::Subst;
use rustc::mir::*;
use rustc::mir::visit::TyContext;
use rustc::traits::PredicateObligations;

use rustc_data_structures::indexed_vec::Idx;

use super::{AtLocation, TypeChecker};

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    pub(super) fn equate_inputs_and_outputs(
        &mut self,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
        universal_regions: &UniversalRegions<'tcx>,
    ) {
        let tcx = self.infcx.tcx;

        let &UniversalRegions {
            unnormalized_output_ty,
            unnormalized_input_tys,
            ..
        } = universal_regions;
        let infcx = self.infcx;

        let start_position = Location {
            block: START_BLOCK,
            statement_index: 0,
        };

        // Equate expected input tys with those in the MIR.
        let argument_locals = (1..).map(Local::new);
        for (&unnormalized_input_ty, local) in unnormalized_input_tys.iter().zip(argument_locals) {
            let input_ty = self.normalize(&unnormalized_input_ty, start_position);
            let mir_input_ty = mir.local_decls[local].ty;
            self.equate_normalized_input_or_output(start_position, input_ty, mir_input_ty);
        }

        // Return types are a bit more complex. They may contain existential `impl Trait`
        // types.
        debug!(
            "equate_inputs_and_outputs: unnormalized_output_ty={:?}",
            unnormalized_output_ty
        );
        let output_ty = self.normalize(&unnormalized_output_ty, start_position);
        debug!(
            "equate_inputs_and_outputs: normalized output_ty={:?}",
            output_ty
        );
        let mir_output_ty = mir.local_decls[RETURN_PLACE].ty;
        let anon_type_map = self.fully_perform_op(start_position.at_self(), |cx| {
            let mut obligations = ObligationAccumulator::default();

            let (output_ty, anon_type_map) = obligations.add(infcx.instantiate_anon_types(
                mir_def_id,
                cx.body_id,
                cx.param_env,
                &output_ty,
            ));
            debug!(
                "equate_inputs_and_outputs: instantiated output_ty={:?}",
                output_ty
            );
            debug!(
                "equate_inputs_and_outputs: anon_type_map={:#?}",
                anon_type_map
            );

            debug!(
                "equate_inputs_and_outputs: mir_output_ty={:?}",
                mir_output_ty
            );
            obligations.add(infcx
                .at(&cx.misc(cx.last_span), cx.param_env)
                .eq(output_ty, mir_output_ty)?);

            for (&anon_def_id, anon_decl) in &anon_type_map {
                let anon_defn_ty = tcx.type_of(anon_def_id);
                let anon_defn_ty = anon_defn_ty.subst(tcx, anon_decl.substs);
                let anon_defn_ty = renumber::renumber_regions(
                    cx.infcx,
                    TyContext::Location(start_position),
                    &anon_defn_ty,
                );
                debug!(
                    "equate_inputs_and_outputs: concrete_ty={:?}",
                    anon_decl.concrete_ty
                );
                debug!("equate_inputs_and_outputs: anon_defn_ty={:?}", anon_defn_ty);
                obligations.add(infcx
                    .at(&cx.misc(cx.last_span), cx.param_env)
                    .eq(anon_decl.concrete_ty, anon_defn_ty)?);
            }

            debug!("equate_inputs_and_outputs: equated");

            Ok(InferOk {
                value: Some(anon_type_map),
                obligations: obligations.into_vec(),
            })
        }).unwrap_or_else(|terr| {
                span_mirbug!(
                    self,
                    start_position,
                    "equate_inputs_and_outputs: `{:?}=={:?}` failed with `{:?}`",
                    output_ty,
                    mir_output_ty,
                    terr
                );
                None
            });

        // Finally, if we instantiated the anon types successfully, we
        // have to solve any bounds (e.g., `-> impl Iterator` needs to
        // prove that `T: Iterator` where `T` is the type we
        // instantiated it with).
        if let Some(anon_type_map) = anon_type_map {
            self.fully_perform_op(start_position.at_self(), |_cx| {
                infcx.constrain_anon_types(&anon_type_map, universal_regions);
                Ok(InferOk {
                    value: (),
                    obligations: vec![],
                })
            }).unwrap();
        }
    }

    fn equate_normalized_input_or_output(&mut self, location: Location, a: Ty<'tcx>, b: Ty<'tcx>) {
        debug!("equate_normalized_input_or_output(a={:?}, b={:?})", a, b);

        if let Err(terr) = self.eq_types(a, b, location.at_self()) {
            span_mirbug!(
                self,
                location,
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
