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

use borrow_check::nll::universal_regions::UniversalRegions;
use rustc::ty::Ty;
use rustc::mir::*;

use rustc_data_structures::indexed_vec::Idx;

use super::{AtLocation, TypeChecker};

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    pub(super) fn equate_inputs_and_outputs(
        &mut self,
        mir: &Mir<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
    ) {
        let &UniversalRegions {
            unnormalized_output_ty,
            unnormalized_input_tys,
            ..
        } = universal_regions;

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

        let output_ty = self.normalize(&unnormalized_output_ty, start_position);
        let mir_output_ty = mir.local_decls[RETURN_PLACE].ty;
        self.equate_normalized_input_or_output(start_position, output_ty, mir_output_ty);
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
