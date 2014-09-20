// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty;
use middle::typeck::infer::{InferCtxt, skolemize};
use util::nodemap::DefIdMap;
use util::ppaux::Repr;

use super::CodeAmbiguity;
use super::Obligation;
use super::FulfillmentError;
use super::CodeSelectionError;
use super::select::SelectionContext;
use super::Unimplemented;

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

    // For semi-hacky reasons (see FIXME below) we keep the builtin
    // trait obligations segregated.
    builtin_obligations: Vec<Obligation>,
}

impl FulfillmentContext {
    pub fn new() -> FulfillmentContext {
        FulfillmentContext {
            trait_obligations: Vec::new(),
            builtin_obligations: Vec::new()
        }
    }

    pub fn register_obligation(&mut self,
                               tcx: &ty::ctxt,
                               obligation: Obligation)
    {
        debug!("register_obligation({})", obligation.repr(tcx));
        match tcx.lang_items.to_builtin_kind(obligation.trait_ref.def_id) {
            Some(_) => {
                self.builtin_obligations.push(obligation);
            }
            None => {
                self.trait_obligations.push(obligation);
            }
        }
    }

    pub fn select_all_or_error(&mut self,
                               infcx: &InferCtxt,
                               param_env: &ty::ParameterEnvironment,
                               unboxed_closures: &DefIdMap<ty::UnboxedClosure>)
                               -> Result<(),Vec<FulfillmentError>>
    {
        try!(self.select_where_possible(infcx, param_env,
                                        unboxed_closures));

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

    pub fn select_where_possible(&mut self,
                                 infcx: &InferCtxt,
                                 param_env: &ty::ParameterEnvironment,
                                 unboxed_closures: &DefIdMap<ty::UnboxedClosure>)
                                 -> Result<(),Vec<FulfillmentError>>
    {
        let tcx = infcx.tcx;
        let selcx = SelectionContext::new(infcx, param_env,
                                          unboxed_closures);

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

    pub fn check_builtin_bound_obligations(
        &self,
        infcx: &InferCtxt)
        -> Result<(),Vec<FulfillmentError>>
    {
        let tcx = infcx.tcx;
        let mut errors = Vec::new();
        debug!("check_builtin_bound_obligations");
        for obligation in self.builtin_obligations.iter() {
            debug!("obligation={}", obligation.repr(tcx));

            let def_id = obligation.trait_ref.def_id;
            let bound = match tcx.lang_items.to_builtin_kind(def_id) {
                Some(bound) => { bound }
                None => { continue; }
            };

            let unskol_self_ty = obligation.self_ty();

            // Skolemize the self-type so that it no longer contains
            // inference variables. Note that this also replaces
            // regions with 'static. You might think that this is not
            // ok, because checking whether something is `Send`
            // implies checking whether it is 'static: that's true,
            // but in fact the region bound is fed into region
            // inference separately and enforced there (and that has
            // even already been done before this code executes,
            // generally speaking).
            let self_ty = skolemize(infcx, unskol_self_ty);

            debug!("bound={} self_ty={}", bound, self_ty.repr(tcx));
            if ty::type_is_error(self_ty) {
                // Indicates an error that was/will-be
                // reported elsewhere.
                continue;
            }

            // Determine if builtin bound is met.
            let tc = ty::type_contents(tcx, self_ty);
            debug!("tc={}", tc);
            let met = match bound {
                ty::BoundSend   => tc.is_sendable(tcx),
                ty::BoundSized  => tc.is_sized(tcx),
                ty::BoundCopy   => tc.is_copy(tcx),
                ty::BoundSync   => tc.is_sync(tcx),
            };

            if met {
                continue;
            }

            // FIXME -- This is kind of a hack: it requently happens
            // that some earlier error prevents types from being fully
            // inferred, and then we get a bunch of uninteresting
            // errors saying something like "<generic #0> doesn't
            // implement Sized".  It may even be true that we could
            // just skip over all checks where the self-ty is an
            // inference variable, but I was afraid that there might
            // be an inference variable created, registered as an
            // obligation, and then never forced by writeback, and
            // hence by skipping here we'd be ignoring the fact that
            // we don't KNOW the type works out. Though even that
            // would probably be harmless, given that we're only
            // talking about builtin traits, which are known to be
            // inhabited. But in any case I just threw in this check
            // for has_errors() to be sure that compilation isn't
            // happening anyway. In that case, why inundate the user.
            if ty::type_needs_infer(self_ty) &&
                tcx.sess.has_errors()
            {
                debug!("skipping printout because self_ty={}",
                       self_ty.repr(tcx));
                continue;
            }

            errors.push(
                FulfillmentError::new(
                    (*obligation).clone(),
                    CodeSelectionError(Unimplemented)));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
