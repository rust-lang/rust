// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::mem_categorization::Typer;
use middle::ty;
use middle::typeck::infer::InferCtxt;
use util::ppaux::Repr;

use super::CodeAmbiguity;
use super::Obligation;
use super::FulfillmentError;
use super::CodeSelectionError;
use super::select::SelectionContext;

/**
 * The fulfillment context is used to drive trait resolution.  It
 * consists of a list of obligations that must be (eventually)
 * satisfied. The job is to track which are satisfied, which yielded
 * errors, and which are still pending. At any point, users can call
 * `select_where_possible`, and the fulfilment context will try to do
 * selection, retaining only those obligations that remain
 * ambiguous. This may be helpful in pushing type inference
 * along. Once all type inference constraints have been generated, the
 * method `select_all_or_error` can be used to report any remaining
 * ambiguous cases as errors.
 */
pub struct FulfillmentContext {
    // A list of all obligations that have been registered with this
    // fulfillment context.
    trait_obligations: Vec<Obligation>,
}

impl FulfillmentContext {
    pub fn new() -> FulfillmentContext {
        FulfillmentContext {
            trait_obligations: Vec::new(),
        }
    }

    pub fn register_obligation(&mut self,
                               tcx: &ty::ctxt,
                               obligation: Obligation)
    {
        debug!("register_obligation({})", obligation.repr(tcx));
        self.trait_obligations.push(obligation);
    }

    pub fn select_all_or_error<'a,'tcx>(&mut self,
                                        infcx: &InferCtxt<'a,'tcx>,
                                        param_env: &ty::ParameterEnvironment,
                                        typer: &Typer<'tcx>)
                                        -> Result<(),Vec<FulfillmentError>>
    {
        try!(self.select_where_possible(infcx, param_env, typer));

        // Anything left is ambiguous.
        let errors: Vec<FulfillmentError> =
            self.trait_obligations
            .iter()
            .map(|o| FulfillmentError::new((*o).clone(), CodeAmbiguity))
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn select_where_possible<'a,'tcx>(&mut self,
                                          infcx: &InferCtxt<'a,'tcx>,
                                          param_env: &ty::ParameterEnvironment,
                                          typer: &Typer<'tcx>)
                                          -> Result<(),Vec<FulfillmentError>>
    {
        let tcx = infcx.tcx;
        let mut selcx = SelectionContext::new(infcx, param_env, typer);

        debug!("select_where_possible({} obligations) start",
               self.trait_obligations.len());

        let mut errors = Vec::new();

        loop {
            let count = self.trait_obligations.len();

            debug!("select_where_possible({} obligations) iteration",
                   count);

            let mut selections = Vec::new();

            // First pass: walk each obligation, retaining
            // only those that we cannot yet process.
            self.trait_obligations.retain(|obligation| {
                match selcx.select(obligation) {
                    Ok(None) => {
                        true
                    }
                    Ok(Some(s)) => {
                        selections.push(s);
                        false
                    }
                    Err(selection_err) => {
                        debug!("obligation: {} error: {}",
                               obligation.repr(tcx),
                               selection_err.repr(tcx));

                        errors.push(FulfillmentError::new(
                            (*obligation).clone(),
                            CodeSelectionError(selection_err)));
                        false
                    }
                }
            });

            if self.trait_obligations.len() == count {
                // Nothing changed.
                break;
            }

            // Now go through all the successful ones,
            // registering any nested obligations for the future.
            for selection in selections.into_iter() {
                selection.map_move_nested(
                    |o| self.register_obligation(tcx, o));
            }
        }

        debug!("select_where_possible({} obligations, {} errors) done",
               self.trait_obligations.len(),
               errors.len());

        if errors.len() == 0 {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
