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
//! `RETURN_PLACE` the MIR arguments) are always fully normalized (and
//! contain revealed `impl Trait` values).

use borrow_check::nll::universal_regions::UniversalRegions;
use rustc::mir::*;
use rustc::ty::Ty;

use rustc_data_structures::indexed_vec::Idx;
use syntax_pos::Span;

use super::{ConstraintCategory, Locations, TypeChecker};

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    pub(super) fn equate_inputs_and_outputs(
        &mut self,
        mir: &Mir<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
        normalized_inputs_and_output: &[Ty<'tcx>],
    ) {
        let (&normalized_output_ty, normalized_input_tys) =
            normalized_inputs_and_output.split_last().unwrap();

        // Equate expected input tys with those in the MIR.
        let argument_locals = (1..).map(Local::new);
        for (&normalized_input_ty, local) in normalized_input_tys.iter().zip(argument_locals) {
            debug!(
                "equate_inputs_and_outputs: normalized_input_ty = {:?}",
                normalized_input_ty
            );

            let mir_input_ty = mir.local_decls[local].ty;
            let mir_input_span = mir.local_decls[local].source_info.span;
            self.equate_normalized_input_or_output(
                normalized_input_ty,
                mir_input_ty,
                mir_input_span,
            );
        }

        assert!(
            mir.yield_ty.is_some() && universal_regions.yield_ty.is_some()
                || mir.yield_ty.is_none() && universal_regions.yield_ty.is_none()
        );
        if let Some(mir_yield_ty) = mir.yield_ty {
            let ur_yield_ty = universal_regions.yield_ty.unwrap();
            let yield_span = mir.local_decls[RETURN_PLACE].source_info.span;
            self.equate_normalized_input_or_output(ur_yield_ty, mir_yield_ty, yield_span);
        }

        // Return types are a bit more complex. They may contain existential `impl Trait`
        // types.
        let mir_output_ty = mir.local_decls[RETURN_PLACE].ty;
        let output_span = mir.local_decls[RETURN_PLACE].source_info.span;
        if let Err(terr) = self.eq_opaque_type_and_type(
            mir_output_ty,
            normalized_output_ty,
            Locations::All(output_span),
            ConstraintCategory::BoringNoLocation,
        ) {
            span_mirbug!(
                self,
                Location::START,
                "equate_inputs_and_outputs: `{:?}=={:?}` failed with `{:?}`",
                normalized_output_ty,
                mir_output_ty,
                terr
            );
        };
    }

    fn equate_normalized_input_or_output(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, span: Span) {
        debug!("equate_normalized_input_or_output(a={:?}, b={:?})", a, b);

        if let Err(terr) = self.eq_types(
            a,
            b,
            Locations::All(span),
            ConstraintCategory::BoringNoLocation,
        ) {
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
