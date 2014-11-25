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
use middle::infer::InferCtxt;
use util::ppaux::Repr;

use super::CodeAmbiguity;
use super::Obligation;
use super::FulfillmentError;
use super::CodeSelectionError;
use super::select::SelectionContext;

/// The fulfillment context is used to drive trait resolution.  It
/// consists of a list of obligations that must be (eventually)
/// satisfied. The job is to track which are satisfied, which yielded
/// errors, and which are still pending. At any point, users can call
/// `select_where_possible`, and the fulfilment context will try to do
/// selection, retaining only those obligations that remain
/// ambiguous. This may be helpful in pushing type inference
/// along. Once all type inference constraints have been generated, the
/// method `select_all_or_error` can be used to report any remaining
/// ambiguous cases as errors.
pub struct FulfillmentContext<'tcx> {
    // A list of all obligations that have been registered with this
    // fulfillment context.
    trait_obligations: Vec<Obligation<'tcx>>,

    // Remembers the count of trait obligations that we have already
    // attempted to select. This is used to avoid repeating work
    // when `select_new_obligations` is called.
    attempted_mark: uint,
}

impl<'tcx> FulfillmentContext<'tcx> {
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            trait_obligations: Vec::new(),
            attempted_mark: 0,
        }
    }

    pub fn register_obligation(&mut self,
                               tcx: &ty::ctxt<'tcx>,
                               obligation: Obligation<'tcx>)
    {
        debug!("register_obligation({})", obligation.repr(tcx));
        assert!(!obligation.trait_ref.has_escaping_regions());
        self.trait_obligations.push(obligation);
    }

    pub fn select_all_or_error<'a>(&mut self,
                                   infcx: &InferCtxt<'a,'tcx>,
                                   param_env: &ty::ParameterEnvironment<'tcx>,
                                   typer: &Typer<'tcx>)
                                   -> Result<(),Vec<FulfillmentError<'tcx>>>
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

    /// Attempts to select obligations that were registered since the call to a selection routine.
    /// This is used by the type checker to eagerly attempt to resolve obligations in hopes of
    /// gaining type information. It'd be equally valid to use `select_where_possible` but it
    /// results in `O(n^2)` performance (#18208).
    pub fn select_new_obligations<'a>(&mut self,
                                      infcx: &InferCtxt<'a,'tcx>,
                                      param_env: &ty::ParameterEnvironment<'tcx>,
                                      typer: &Typer<'tcx>)
                                      -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        let mut selcx = SelectionContext::new(infcx, param_env, typer);
        self.select(&mut selcx, true)
    }

    pub fn select_where_possible<'a>(&mut self,
                                     infcx: &InferCtxt<'a,'tcx>,
                                     param_env: &ty::ParameterEnvironment<'tcx>,
                                     typer: &Typer<'tcx>)
                                     -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        let mut selcx = SelectionContext::new(infcx, param_env, typer);
        self.select(&mut selcx, false)
    }

    pub fn pending_trait_obligations(&self) -> &[Obligation<'tcx>] {
        self.trait_obligations[]
    }

    /// Attempts to select obligations using `selcx`. If `only_new_obligations` is true, then it
    /// only attempts to select obligations that haven't been seen before.
    fn select<'a>(&mut self,
                  selcx: &mut SelectionContext<'a, 'tcx>,
                  only_new_obligations: bool)
                  -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        debug!("select({} obligations, only_new_obligations={}) start",
               self.trait_obligations.len(),
               only_new_obligations);

        let tcx = selcx.tcx();
        let mut errors = Vec::new();

        loop {
            let count = self.trait_obligations.len();

            debug!("select_where_possible({} obligations) iteration",
                   count);

            let mut selections = Vec::new();

            // If we are only attempting obligations we haven't seen yet,
            // then set `skip` to the number of obligations we've already
            // seen.
            let mut skip = if only_new_obligations {
                self.attempted_mark
            } else {
                0
            };

            // First pass: walk each obligation, retaining
            // only those that we cannot yet process.
            self.trait_obligations.retain(|obligation| {
                // Hack: Retain does not pass in the index, but we want
                // to avoid processing the first `start_count` entries.
                if skip > 0 {
                    skip -= 1;
                    true
                } else {
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
                }
            });

            self.attempted_mark = self.trait_obligations.len();

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

        debug!("select({} obligations, {} errors) done",
               self.trait_obligations.len(),
               errors.len());

        if errors.len() == 0 {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
