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
use middle::ty::{mod, Ty};
use middle::infer::{mod, InferCtxt};
use std::collections::HashSet;
use std::collections::hash_map::{Occupied, Vacant};
use std::default::Default;
use std::rc::Rc;
use syntax::ast;
use util::ppaux::Repr;
use util::nodemap::NodeMap;

use super::CodeAmbiguity;
use super::TraitObligation;
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
    // a simple cache that aims to cache *exact duplicate obligations*
    // and avoid adding them twice. This serves a different purpose
    // than the `SelectionCache`: it avoids duplicate errors and
    // permits recursive obligations, which are often generated from
    // traits like `Send` et al.
    duplicate_set: HashSet<Rc<ty::TraitRef<'tcx>>>,

    // A list of all obligations that have been registered with this
    // fulfillment context.
    trait_obligations: Vec<TraitObligation<'tcx>>,

    // Remembers the count of trait obligations that we have already
    // attempted to select. This is used to avoid repeating work
    // when `select_new_obligations` is called.
    attempted_mark: uint,

    // A set of constraints that regionck must validate. Each
    // constraint has the form `T:'a`, meaning "some type `T` must
    // outlive the lifetime 'a". These constraints derive from
    // instantiated type parameters. So if you had a struct defined
    // like
    //
    //     struct Foo<T:'static> { ... }
    //
    // then in some expression `let x = Foo { ... }` it will
    // instantiate the type parameter `T` with a fresh type `$0`. At
    // the same time, it will record a region obligation of
    // `$0:'static`. This will get checked later by regionck. (We
    // can't generally check these things right away because we have
    // to wait until types are resolved.)
    //
    // These are stored in a map keyed to the id of the innermost
    // enclosing fn body / static initializer expression. This is
    // because the location where the obligation was incurred can be
    // relevant with respect to which sublifetime assumptions are in
    // place. The reason that we store under the fn-id, and not
    // something more fine-grained, is so that it is easier for
    // regionck to be sure that it has found *all* the region
    // obligations (otherwise, it's easy to fail to walk to a
    // particular node-id).
    region_obligations: NodeMap<Vec<RegionObligation<'tcx>>>,
}

pub struct RegionObligation<'tcx> {
    pub sub_region: ty::Region,
    pub sup_type: Ty<'tcx>,
    pub origin: infer::SubregionOrigin<'tcx>,
}

impl<'tcx> FulfillmentContext<'tcx> {
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            duplicate_set: HashSet::new(),
            trait_obligations: Vec::new(),
            attempted_mark: 0,
            region_obligations: NodeMap::new(),
        }
    }

    pub fn register_obligation(&mut self,
                               tcx: &ty::ctxt<'tcx>,
                               obligation: TraitObligation<'tcx>)
    {
        if self.duplicate_set.insert(obligation.trait_ref.clone()) {
            debug!("register_obligation({})", obligation.repr(tcx));
            assert!(!obligation.trait_ref.has_escaping_regions());
            self.trait_obligations.push(obligation);
        } else {
            debug!("register_obligation({}) -- already seen, skip", obligation.repr(tcx));
        }
    }

    pub fn register_region_obligation(&mut self,
                                      body_id: ast::NodeId,
                                      region_obligation: RegionObligation<'tcx>)
    {
        match self.region_obligations.entry(body_id) {
            Vacant(entry) => { entry.set(vec![region_obligation]); },
            Occupied(mut entry) => { entry.get_mut().push(region_obligation); },
        }
    }

    pub fn region_obligations(&self,
                              body_id: ast::NodeId)
                              -> &[RegionObligation<'tcx>]
    {
        match self.region_obligations.get(&body_id) {
            None => Default::default(),
            Some(vec) => vec.as_slice(),
        }
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

    pub fn pending_trait_obligations(&self) -> &[TraitObligation<'tcx>] {
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

impl<'tcx> Repr<'tcx> for RegionObligation<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("RegionObligation(sub_region={}, sup_type={}, origin={})",
                self.sub_region.repr(tcx),
                self.sup_type.repr(tcx),
                self.origin.repr(tcx))
    }
}
