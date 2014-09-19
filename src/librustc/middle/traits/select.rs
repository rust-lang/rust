// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See `doc.rs` for high-level documentation */

use super::{Obligation, ObligationCause};
use super::{EvaluationResult, EvaluatedToMatch,
            EvaluatedToAmbiguity, EvaluatedToUnmatch};
use super::{SelectionError, Unimplemented, Overflow,
            OutputTypeParameterMismatch};
use super::{Selection};
use super::{SelectionResult};
use super::{VtableBuiltin, VtableImpl, VtableParam, VtableUnboxedClosure};
use super::{VtableImplData, VtableParamData};
use super::{util};

use middle::subst::{Subst, Substs, VecPerParamSpace};
use middle::ty;
use middle::typeck::check::regionmanip;
use middle::typeck::infer;
use middle::typeck::infer::InferCtxt;
use std::rc::Rc;
use syntax::ast;
use util::nodemap::DefIdMap;
use util::ppaux::Repr;

pub struct SelectionContext<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    param_env: &'cx ty::ParameterEnvironment,
    unboxed_closures: &'cx DefIdMap<ty::UnboxedClosure>,
}

// pub struct SelectionCache {
//     hashmap: RefCell<HashMap<CacheKey, Candidate>>,
// }

// #[deriving(Hash,Eq,PartialEq)]
// struct CacheKey {
//     trait_def_id: ast::DefId,
//     skol_obligation_self_ty: ty::t,
// }

enum MatchResult<T> {
    Matched(T),
    AmbiguousMatch,
    NoMatch
}

/**
 * The selection process begins by considering all impls, where
 * clauses, and so forth that might resolve an obligation.  Sometimes
 * we'll be able to say definitively that (e.g.) an impl does not
 * apply to the obligation: perhaps it is defined for `uint` but the
 * obligation is for `int`. In that case, we drop the impl out of the
 * list.  But the other cases are considered *candidates*.
 *
 * Candidates can either be definitive or ambiguous. An ambiguous
 * candidate is one that might match or might not, depending on how
 * type variables wind up being resolved. This only occurs during inference.
 *
 * For selection to suceed, there must be exactly one non-ambiguous
 * candidate.  Usually, it is not possible to have more than one
 * definitive candidate, due to the coherence rules. However, there is
 * one case where it could occur: if there is a blanket impl for a
 * trait (that is, an impl applied to all T), and a type parameter
 * with a where clause. In that case, we can have a candidate from the
 * where clause and a second candidate from the impl. This is not a
 * problem because coherence guarantees us that the impl which would
 * be used to satisfy the where clause is the same one that we see
 * now. To resolve this issue, therefore, we ignore impls if we find a
 * matching where clause. Part of the reason for this is that where
 * clauses can give additional information (like, the types of output
 * parameters) that would have to be inferred from the impl.
 */
#[deriving(Clone)]
enum Candidate {
    MatchedBuiltinCandidate,
    AmbiguousBuiltinCandidate,
    MatchedParamCandidate(VtableParamData),
    AmbiguousParamCandidate,
    Impl(ImplCandidate),
    MatchedUnboxedClosureCandidate(/* closure */ ast::DefId)
}

#[deriving(Clone)]
enum ImplCandidate {
    MatchedImplCandidate(ast::DefId),
    AmbiguousImplCandidate(ast::DefId),
}

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    pub fn new(infcx: &'cx InferCtxt<'cx, 'tcx>,
               param_env: &'cx ty::ParameterEnvironment,
               unboxed_closures: &'cx DefIdMap<ty::UnboxedClosure>)
               -> SelectionContext<'cx, 'tcx> {
        SelectionContext { infcx: infcx, param_env: param_env,
                           unboxed_closures: unboxed_closures }
    }

    pub fn tcx(&self) -> &'cx ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    ///////////////////////////////////////////////////////////////////////////
    // Selection
    //
    // The selection phase tries to identify *how* an obligation will
    // be resolved. For example, it will identify which impl or
    // parameter bound is to be used. The process can be inconclusive
    // if the self type in the obligation is not fully inferred. Selection
    // can result in an error in one of two ways:
    //
    // 1. If no applicable impl or parameter bound can be found.
    // 2. If the output type parameters in the obligation do not match
    //    those specified by the impl/bound. For example, if the obligation
    //    is `Vec<Foo>:Iterable<Bar>`, but the impl specifies
    //    `impl<T> Iterable<T> for Vec<T>`, than an error would result.

    pub fn select(&self, obligation: &Obligation) -> SelectionResult<Selection> {
        /*!
         * Evaluates whether the obligation can be satisfied. Returns
         * an indication of whether the obligation can be satisfied
         * and, if so, by what means. Never affects surrounding typing
         * environment.
         */

        debug!("select({})", obligation.repr(self.tcx()));

        match try!(self.candidate_from_obligation(obligation)) {
            None => Ok(None),
            Some(candidate) => self.confirm_candidate(obligation, candidate),
        }
    }

    pub fn select_inherent_impl(&self,
                                impl_def_id: ast::DefId,
                                obligation_cause: ObligationCause,
                                obligation_self_ty: ty::t)
                                -> SelectionResult<VtableImplData<Obligation>>
    {
        debug!("select_inherent_impl(impl_def_id={}, obligation_self_ty={})",
               impl_def_id.repr(self.tcx()),
               obligation_self_ty.repr(self.tcx()));

        match self.candidate_from_impl(impl_def_id,
                                       obligation_cause,
                                       obligation_self_ty) {
            Some(MatchedImplCandidate(impl_def_id)) => {
                let vtable_impl =
                    try!(self.confirm_inherent_impl_candidate(
                        impl_def_id,
                        obligation_cause,
                        obligation_self_ty,
                        0));
                Ok(Some(vtable_impl))
            }
            Some(AmbiguousImplCandidate(_)) => {
                Ok(None)
            }
            None => {
                Err(Unimplemented)
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // EVALUATION
    //
    // Tests whether an obligation can be selected or whether an impl can be
    // applied to particular types. It skips the "confirmation" step and
    // hence completely ignores output type parameters.

    pub fn evaluate_obligation(&self,
                               obligation: &Obligation)
                               -> EvaluationResult
    {
        /*!
         * Evaluates whether the obligation `obligation` can be
         * satisfied (by any means).
         */

        debug!("evaluate_obligation({})",
               obligation.repr(self.tcx()));

        match self.candidate_from_obligation(obligation) {
            Ok(Some(c)) => c.to_evaluation_result(),
            Ok(None) => EvaluatedToAmbiguity,
            Err(_) => EvaluatedToUnmatch,
        }
    }

    pub fn evaluate_impl(&self,
                         impl_def_id: ast::DefId,
                         obligation_cause: ObligationCause,
                         obligation_self_ty: ty::t)
                         -> EvaluationResult
    {
        /*!
         * Evaluates whether the impl with id `impl_def_id` could be
         * applied to the self type `obligation_self_ty`. This can be
         * used either for trait or inherent impls.
         */

        debug!("evaluate_impl(impl_def_id={}, obligation_self_ty={})",
               impl_def_id.repr(self.tcx()),
               obligation_self_ty.repr(self.tcx()));

        match self.candidate_from_impl(impl_def_id,
                                       obligation_cause,
                                       obligation_self_ty) {
            Some(c) => c.to_evaluation_result(),
            None => EvaluatedToUnmatch,
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY
    //
    // The selection process begins by examining all in-scope impls,
    // caller obligations, and so forth and assembling a list of
    // candidates. See `doc.rs` and the `Candidate` type for more details.

    fn candidate_from_obligation(&self, obligation: &Obligation)
                                 -> SelectionResult<Candidate>
    {
        debug!("candidate_from_obligation({}, self_ty={})",
               obligation.repr(self.tcx()),
               self.infcx.ty_to_string(obligation.self_ty()));

        let skol_obligation_self_ty =
            infer::skolemize(self.infcx, obligation.self_ty());

        // First, check the cache.
        match self.check_candidate_cache(obligation, skol_obligation_self_ty) {
            Some(c) => {
                return Ok(Some(c));
            }
            None => { }
        }

        let mut candidates =
            try!(self.assemble_candidates(obligation,
                                          skol_obligation_self_ty));

        debug!("candidate_from_obligation: {} candidates for {}",
               candidates.len(), obligation.repr(self.tcx()));

        // Examine candidates to determine outcome. Ideally we will
        // have exactly one candidate that is definitively applicable.

        if candidates.len() == 0 {
            // Annoying edge case: if there are no impls, then there
            // is no way that this trait reference is implemented,
            // *unless* it contains unbound variables. In that case,
            // it is possible that one of those unbound variables will
            // be bound to a new type from some other crate which will
            // also contain impls.
            let trait_ref = &*obligation.trait_ref;
            return if !self.trait_ref_unconstrained(trait_ref) {
                debug!("candidate_from_obligation({}) -> 0 matches, unimpl",
                       obligation.repr(self.tcx()));
                Err(Unimplemented)
            } else {
                debug!("candidate_from_obligation({}) -> 0 matches, ambig",
                       obligation.repr(self.tcx()));
                Ok(None)
            };
        }

        if candidates.len() > 1 {
            // Ambiguity. Possibly we should report back more
            // information on the potential candidates so we can give
            // a better error message.
            debug!("candidate_from_obligation({}) -> multiple matches, ambig",
                   obligation.repr(self.tcx()));

            return Ok(None);
        }

        let candidate = candidates.pop().unwrap();
        self.insert_candidate_cache(obligation, skol_obligation_self_ty,
                                    candidate.clone());
        Ok(Some(candidate))
    }

    fn check_candidate_cache(&self,
                             _obligation: &Obligation,
                             _skol_obligation_self_ty: ty::t)
                             -> Option<Candidate>
    {
        // let cache_key = CacheKey::new(obligation.trait_ref.def_id,
        //                               skol_obligation_self_ty);
        // let hashmap = self.tcx().selection_cache.hashmap.borrow();
        // hashmap.find(&cache_key).map(|c| (*c).clone())
        None
    }

    fn insert_candidate_cache(&self,
                              _obligation: &Obligation,
                              _skol_obligation_self_ty: ty::t,
                              _candidate: Candidate)
    {
        // FIXME -- Enable caching. I think the right place to put the cache
        // is in the ParameterEnvironment, not the tcx, because otherwise
        // when there are distinct where clauses in scope the cache can get
        // confused.
        //
        //let cache_key = CacheKey::new(obligation.trait_ref.def_id,
        //                              skol_obligation_self_ty);
        //let mut hashmap = self.tcx().selection_cache.hashmap.borrow_mut();
        //hashmap.insert(cache_key, candidate);
    }

    fn assemble_candidates(&self,
                           obligation: &Obligation,
                           skol_obligation_self_ty: ty::t)
                           -> Result<Vec<Candidate>, SelectionError>
    {
        // Check for overflow.

        let recursion_limit = self.infcx.tcx.sess.recursion_limit.get();
        if obligation.recursion_depth >= recursion_limit {
            debug!("{} --> overflow", obligation.repr(self.tcx()));
            return Err(Overflow);
        }

        let mut candidates = Vec::new();

        match self.tcx().lang_items.to_builtin_kind(obligation.trait_ref.def_id) {
            Some(_) => {
                // FIXME -- The treatment of builtin bounds is a bit
                // hacky right now. Eventually, the idea is to move
                // the logic for selection out of type_contents and
                // into this module (And make it based on the generic
                // mechanisms of OIBTT2).  However, I want to land
                // some code today, so we're going to cut a few
                // corners. What we do now is that the trait selection
                // code always considers builtin obligations to
                // match. The fulfillment code (which also has the job
                // of tracking all the traits that must hold) will
                // then just accumulate the various
                // builtin-bound-related obligations that must be met.
                // Later, at the end of typeck, after writeback etc,
                // we will rewalk this list and extract all the
                // builtin-bound-related obligations and test them
                // again using type contents. Part of the motivation
                // for this is that the type contents code requires
                // that writeback has been completed in some cases.

                candidates.push(AmbiguousBuiltinCandidate);
            }

            None => {
                // Other bounds. Consider both in-scope bounds from fn decl
                // and applicable impls.

                try!(self.assemble_candidates_from_caller_bounds(
                    obligation,
                    skol_obligation_self_ty,
                    &mut candidates));

                try!(self.assemble_unboxed_candidates(
                    obligation,
                    skol_obligation_self_ty,
                    &mut candidates));

                // If there is a fn bound that applies, forego the
                // impl search. It can only generate conflicts.

                if candidates.len() == 0 {
                    try!(self.assemble_candidates_from_impls(
                        obligation,
                        skol_obligation_self_ty,
                        &mut candidates));
                }
            }
        }

        Ok(candidates)
    }

    fn assemble_candidates_from_caller_bounds(&self,
                                              obligation: &Obligation,
                                              skol_obligation_self_ty: ty::t,
                                              candidates: &mut Vec<Candidate>)
                                              -> Result<(),SelectionError>
    {
        /*!
         * Given an obligation like `<SomeTrait for T>`, search the obligations
         * that the caller supplied to find out whether it is listed among
         * them.
         *
         * Never affects inference environment.
v         */

        debug!("assemble_candidates_from_caller_bounds({})",
               obligation.repr(self.tcx()));

        for caller_obligation in self.param_env.caller_obligations.iter() {
            debug!("caller_obligation={}",
                   caller_obligation.repr(self.tcx()));

            // Skip over obligations that don't apply to
            // `self_ty`.
            let caller_bound = &caller_obligation.trait_ref;
            let caller_self_ty = caller_bound.substs.self_ty().unwrap();
            match self.match_self_types(obligation.cause,
                                        caller_self_ty,
                                        skol_obligation_self_ty) {
                AmbiguousMatch => {
                    debug!("-> AmbiguousParamCandidate");
                    candidates.push(AmbiguousParamCandidate);
                    return Ok(());
                }
                NoMatch => {
                    continue;
                }
                Matched(()) => { }
            }

            // Search through the trait (and its supertraits) to
            // see if it matches the def-id we are looking for.
            let caller_bound = (*caller_bound).clone();
            match util::search_trait_and_supertraits_from_bound(
                self.infcx.tcx, caller_bound,
                |d| d == obligation.trait_ref.def_id)
            {
                Some(vtable_param) => {
                    // If so, we're done!
                    debug!("-> MatchedParamCandidate({})", vtable_param);
                    candidates.push(MatchedParamCandidate(vtable_param));
                    return Ok(());
                }

                None => {
                }
            }
        }

        Ok(())
    }

    fn assemble_unboxed_candidates(&self,
                                   obligation: &Obligation,
                                   skol_obligation_self_ty: ty::t,
                                   candidates: &mut Vec<Candidate>)
                                   -> Result<(),SelectionError>
    {
        /*!
         * Check for the artificial impl that the compiler will create
         * for an obligation like `X : FnMut<..>` where `X` is an
         * unboxed closure type.
         */

        let closure_def_id = match ty::get(skol_obligation_self_ty).sty {
            ty::ty_unboxed_closure(id, _) => id,
            _ => { return Ok(()); }
        };

        let tcx = self.tcx();
        let fn_traits = [
            (ty::FnUnboxedClosureKind, tcx.lang_items.fn_trait()),
            (ty::FnMutUnboxedClosureKind, tcx.lang_items.fn_mut_trait()),
            (ty::FnOnceUnboxedClosureKind, tcx.lang_items.fn_once_trait()),
            ];
        for tuple in fn_traits.iter() {
            let kind = match tuple {
                &(kind, Some(ref fn_trait))
                    if *fn_trait == obligation.trait_ref.def_id =>
                {
                    kind
                }
                _ => continue,
            };

            // Check to see whether the argument and return types match.
            let closure_kind = match self.unboxed_closures.find(&closure_def_id) {
                Some(closure) => closure.kind,
                None => {
                    self.tcx().sess.span_bug(
                        obligation.cause.span,
                        format!("No entry for unboxed closure: {}",
                                closure_def_id.repr(self.tcx())).as_slice());
                }
            };

            if closure_kind != kind {
                continue;
            }

            candidates.push(MatchedUnboxedClosureCandidate(closure_def_id));
        }

        Ok(())
    }

    fn assemble_candidates_from_impls(&self,
                                      obligation: &Obligation,
                                      skol_obligation_self_ty: ty::t,
                                      candidates: &mut Vec<Candidate>)
                                      -> Result<(), SelectionError>
    {
        /*!
         * Search for impls that might apply to `obligation`.
         */

        let all_impls = self.all_impls(obligation.trait_ref.def_id);
        for &impl_def_id in all_impls.iter() {
            self.infcx.probe(|| {
                match self.candidate_from_impl(impl_def_id,
                                               obligation.cause,
                                               skol_obligation_self_ty) {
                    Some(c) => {
                        candidates.push(Impl(c));
                    }

                    None => { }
                }
            });
        }
        Ok(())
    }

    fn candidate_from_impl(&self,
                           impl_def_id: ast::DefId,
                           obligation_cause: ObligationCause,
                           skol_obligation_self_ty: ty::t)
                           -> Option<ImplCandidate>
    {
        match self.match_impl_self_types(impl_def_id,
                                         obligation_cause,
                                         skol_obligation_self_ty) {
            Matched(_) => {
                Some(MatchedImplCandidate(impl_def_id))
            }

            AmbiguousMatch => {
                Some(AmbiguousImplCandidate(impl_def_id))
            }

            NoMatch => {
                None
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // CONFIRMATION
    //
    // Confirmation unifies the output type parameters of the trait
    // with the values found in the obligation, possibly yielding a
    // type error.  See `doc.rs` for more details.

    fn confirm_candidate(&self,
                         obligation: &Obligation,
                         candidate: Candidate)
                         -> SelectionResult<Selection>
    {
        debug!("confirm_candidate({}, {})",
               obligation.repr(self.tcx()),
               candidate.repr(self.tcx()));

        match candidate {
            AmbiguousBuiltinCandidate |
            AmbiguousParamCandidate |
            Impl(AmbiguousImplCandidate(_)) => {
                Ok(None)
            }

            MatchedBuiltinCandidate => {
                Ok(Some(VtableBuiltin))
            }

            MatchedParamCandidate(param) => {
                Ok(Some(VtableParam(
                    try!(self.confirm_param_candidate(obligation, param)))))
            }

            Impl(MatchedImplCandidate(impl_def_id)) => {
                let vtable_impl = try!(self.confirm_impl_candidate(obligation,
                                                                   impl_def_id));
                Ok(Some(VtableImpl(vtable_impl)))
            }

            MatchedUnboxedClosureCandidate(closure_def_id) => {
                try!(self.confirm_unboxed_closure_candidate(obligation, closure_def_id));
                Ok(Some(VtableUnboxedClosure(closure_def_id)))
            }
        }
    }

    fn confirm_param_candidate(&self,
                               obligation: &Obligation,
                               param: VtableParamData)
                               -> Result<VtableParamData,SelectionError>
    {
        debug!("confirm_param_candidate({},{})",
               obligation.repr(self.tcx()),
               param.repr(self.tcx()));

        let () = try!(self.confirm(obligation.cause,
                                   obligation.trait_ref.clone(),
                                   param.bound.clone()));
        Ok(param)
    }

    fn confirm_impl_candidate(&self,
                              obligation: &Obligation,
                              impl_def_id: ast::DefId)
                              -> Result<VtableImplData<Obligation>,SelectionError>
    {
        debug!("confirm_impl_candidate({},{})",
               obligation.repr(self.tcx()),
               impl_def_id.repr(self.tcx()));

        // For a non-inhernet impl, we begin the same way as an
        // inherent impl, by matching the self-type and assembling
        // list of nested obligations.
        let vtable_impl =
            try!(self.confirm_inherent_impl_candidate(
                impl_def_id,
                obligation.cause,
                obligation.trait_ref.self_ty(),
                obligation.recursion_depth));

        // But then we must also match the output types.
        let () = try!(self.confirm_impl_vtable(impl_def_id,
                                               obligation.cause,
                                               obligation.trait_ref.clone(),
                                               &vtable_impl.substs));
        Ok(vtable_impl)
    }

    fn confirm_inherent_impl_candidate(&self,
                                       impl_def_id: ast::DefId,
                                       obligation_cause: ObligationCause,
                                       obligation_self_ty: ty::t,
                                       obligation_recursion_depth: uint)
                                       -> Result<VtableImplData<Obligation>,
                                                 SelectionError>
    {
        let substs = match self.match_impl_self_types(impl_def_id,
                                                      obligation_cause,
                                                      obligation_self_ty) {
            Matched(substs) => substs,
            AmbiguousMatch | NoMatch => {
                self.tcx().sess.bug(
                    format!("Impl {} was matchable against {} but now is not",
                            impl_def_id.repr(self.tcx()),
                            obligation_self_ty.repr(self.tcx()))
                        .as_slice());
            }
        };

        let impl_obligations =
            self.impl_obligations(obligation_cause,
                                  obligation_recursion_depth,
                                  impl_def_id,
                                  &substs);
        let vtable_impl = VtableImplData { impl_def_id: impl_def_id,
                                       substs: substs,
                                       nested: impl_obligations };

        Ok(vtable_impl)
    }

    fn confirm_unboxed_closure_candidate(&self,
                                         obligation: &Obligation,
                                         closure_def_id: ast::DefId)
                                         -> Result<(),SelectionError>
    {
        debug!("confirm_unboxed_closure_candidate({},{})",
               obligation.repr(self.tcx()),
               closure_def_id.repr(self.tcx()));

        let closure_type = match self.unboxed_closures.find(&closure_def_id) {
            Some(closure) => closure.closure_type.clone(),
            None => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    format!("No entry for unboxed closure: {}",
                            closure_def_id.repr(self.tcx())).as_slice());
            }
        };

        // FIXME(pcwalton): This is a bogus thing to do, but
        // it'll do for now until we get the new trait-bound
        // region skolemization working.
        let (_, new_signature) =
            regionmanip::replace_late_bound_regions_in_fn_sig(
                self.tcx(),
                &closure_type.sig,
                |br| self.infcx.next_region_var(
                         infer::LateBoundRegion(obligation.cause.span, br)));

        let arguments_tuple = *new_signature.inputs.get(0);
        let trait_ref = Rc::new(ty::TraitRef {
            def_id: obligation.trait_ref.def_id,
            substs: Substs::new_trait(
                vec![arguments_tuple, new_signature.output],
                vec![],
                obligation.self_ty())
        });

        self.confirm(obligation.cause,
                     obligation.trait_ref.clone(),
                     trait_ref)
    }

    ///////////////////////////////////////////////////////////////////////////
    // Matching
    //
    // Matching is a common path used for both evaluation and
    // confirmation.  It basically unifies types that appear in impls
    // and traits. This does affect the surrounding environment;
    // therefore, when used during evaluation, match routines must be
    // run inside of a `probe()` so that their side-effects are
    // contained.

    fn match_impl_self_types(&self,
                             impl_def_id: ast::DefId,
                             obligation_cause: ObligationCause,
                             obligation_self_ty: ty::t)
                             -> MatchResult<Substs>
    {
        /*!
         * Determines whether the self type declared against
         * `impl_def_id` matches `obligation_self_ty`. If successful,
         * returns the substitutions used to make them match. See
         * `match_impl()`.  For example, if `impl_def_id` is declared
         * as:
         *
         *    impl<T:Copy> Foo for ~T { ... }
         *
         * and `obligation_self_ty` is `int`, we'd back an `Err(_)`
         * result. But if `obligation_self_ty` were `~int`, we'd get
         * back `Ok(T=int)`.
         */

        // Create fresh type variables for each type parameter declared
        // on the impl etc.
        let impl_substs = util::fresh_substs_for_impl(self.infcx,
                                                      obligation_cause.span,
                                                      impl_def_id);

        // Find the self type for the impl.
        let impl_self_ty = ty::lookup_item_type(self.tcx(), impl_def_id).ty;
        let impl_self_ty = impl_self_ty.subst(self.tcx(), &impl_substs);

        debug!("match_impl_self_types(obligation_self_ty={}, impl_self_ty={})",
               obligation_self_ty.repr(self.tcx()),
               impl_self_ty.repr(self.tcx()));

        match self.match_self_types(obligation_cause,
                                    impl_self_ty,
                                    obligation_self_ty) {
            Matched(()) => {
                debug!("Matched impl_substs={}", impl_substs.repr(self.tcx()));
                Matched(impl_substs)
            }
            AmbiguousMatch => {
                debug!("AmbiguousMatch");
                AmbiguousMatch
            }
            NoMatch => {
                debug!("NoMatch");
                NoMatch
            }
        }
    }

    fn match_self_types(&self,
                        cause: ObligationCause,

                        // The self type provided by the impl/caller-obligation:
                        provided_self_ty: ty::t,

                        // The self type the obligation is for:
                        required_self_ty: ty::t)
                        -> MatchResult<()>
    {
        // FIXME(#5781) -- equating the types is stronger than
        // necessary. Should consider variance of trait w/r/t Self.

        let origin = infer::RelateSelfType(cause.span);
        match self.infcx.eq_types(false,
                                  origin,
                                  provided_self_ty,
                                  required_self_ty) {
            Ok(()) => Matched(()),
            Err(ty::terr_sorts(ty::expected_found{expected: t1, found: t2})) => {
                // This error occurs when there is an unresolved type
                // variable in the `required_self_ty` that was forced
                // to unify with a non-type-variable. That basically
                // means we don't know enough to say with certainty
                // whether there is a match or not -- it depends on
                // how that type variable is ultimately resolved.
                if ty::type_is_skolemized(t1) || ty::type_is_skolemized(t2) {
                    AmbiguousMatch
                } else {
                    NoMatch
                }
            }
            Err(_) => NoMatch,
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Confirmation
    //
    // The final step of selection: once we know how an obligation is
    // is resolved, we confirm that selection in order to have
    // side-effects on the typing environment. This step also unifies
    // the output type parameters from the obligation with those found
    // on the impl/bound, which may yield type errors.

    fn confirm_impl_vtable(&self,
                           impl_def_id: ast::DefId,
                           obligation_cause: ObligationCause,
                           obligation_trait_ref: Rc<ty::TraitRef>,
                           substs: &Substs)
                           -> Result<(), SelectionError>
    {
        /*!
         * Relates the output type parameters from an impl to the
         * trait.  This may lead to type errors. The confirmation step
         * is separated from the main match procedure because these
         * type errors do not cause us to select another impl.
         *
         * As an example, consider matching the obligation
         * `Iterator<char> for Elems<int>` using the following impl:
         *
         *    impl<T> Iterator<T> for Elems<T> { ... }
         *
         * The match phase will succeed with substitution `T=int`.
         * The confirm step will then try to unify `int` and `char`
         * and yield an error.
         */

        let impl_trait_ref = ty::impl_trait_ref(self.tcx(),
                                                impl_def_id).unwrap();
        let impl_trait_ref = impl_trait_ref.subst(self.tcx(),
                                                  substs);
        self.confirm(obligation_cause, obligation_trait_ref, impl_trait_ref)
    }

    fn confirm(&self,
               obligation_cause: ObligationCause,
               obligation_trait_ref: Rc<ty::TraitRef>,
               expected_trait_ref: Rc<ty::TraitRef>)
               -> Result<(), SelectionError>
    {
        /*!
         * After we have determined which impl applies, and with what
         * substitutions, there is one last step. We have to go back
         * and relate the "output" type parameters from the obligation
         * to the types that are specified in the impl.
         *
         * For example, imagine we have:
         *
         *     impl<T> Iterator<T> for Vec<T> { ... }
         *
         * and our obligation is `Iterator<Foo> for Vec<int>` (note
         * the mismatch in the obligation types). Up until this step,
         * no error would be reported: the self type is `Vec<int>`,
         * and that matches `Vec<T>` with the substitution `T=int`.
         * At this stage, we could then go and check that the type
         * parameters to the `Iterator` trait match.
         * (In terms of the parameters, the `expected_trait_ref`
         * here would be `Iterator<int> for Vec<int>`, and the
         * `obligation_trait_ref` would be `Iterator<Foo> for Vec<int>`.
         *
         * Note that this checking occurs *after* the impl has
         * selected, because these output type parameters should not
         * affect the selection of the impl. Therefore, if there is a
         * mismatch, we report an error to the user.
         */

        let origin = infer::RelateOutputImplTypes(obligation_cause.span);

        let obligation_trait_ref = obligation_trait_ref.clone();
        match self.infcx.sub_trait_refs(false,
                                        origin,
                                        expected_trait_ref.clone(),
                                        obligation_trait_ref) {
            Ok(()) => Ok(()),
            Err(e) => Err(OutputTypeParameterMismatch(expected_trait_ref, e))
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Miscellany

    fn all_impls(&self, trait_def_id: ast::DefId) -> Vec<ast::DefId> {
        /*!
         * Returns se tof all impls for a given trait.
         */

        ty::populate_implementations_for_trait_if_necessary(self.tcx(),
                                                            trait_def_id);
        match self.tcx().trait_impls.borrow().find(&trait_def_id) {
            None => Vec::new(),
            Some(impls) => impls.borrow().clone()
        }
    }

    fn impl_obligations(&self,
                        cause: ObligationCause,
                        recursion_depth: uint,
                        impl_def_id: ast::DefId,
                        impl_substs: &Substs)
                        -> VecPerParamSpace<Obligation>
    {
        let impl_generics = ty::lookup_item_type(self.tcx(),
                                                 impl_def_id).generics;
        util::obligations_for_generics(self.tcx(), cause, recursion_depth,
                                       &impl_generics, impl_substs)
    }

    fn trait_ref_unconstrained(&self,
                               trait_ref: &ty::TraitRef)
                               -> bool
    {
        /*!
         * True if the self type of the trait-ref contains
         * unconstrained type variables.
         */

        let mut found_skol = false;

        // Skolemization replaces all unconstrained type vars with
        // a SkolemizedTy instance. Then we search to see if we
        // found any.
        let skol_ty = infer::skolemize(self.infcx, trait_ref.self_ty());
        ty::walk_ty(skol_ty, |t| {
            match ty::get(t).sty {
                ty::ty_infer(ty::SkolemizedTy(_)) => { found_skol = true; }
                _ => { }
            }
        });

        found_skol
    }
}

impl Candidate {
    fn to_evaluation_result(&self) -> EvaluationResult {
        match *self {
            Impl(ref i) => i.to_evaluation_result(),

            MatchedUnboxedClosureCandidate(..) |
            MatchedBuiltinCandidate |
            MatchedParamCandidate(..) => {
                EvaluatedToMatch
            }

            AmbiguousBuiltinCandidate |
            AmbiguousParamCandidate => {
                EvaluatedToAmbiguity
            }
        }
    }
}

impl ImplCandidate {
    fn to_evaluation_result(&self) -> EvaluationResult {
        match *self {
            MatchedImplCandidate(..) => EvaluatedToMatch,
            AmbiguousImplCandidate(..) => EvaluatedToAmbiguity
        }
    }
}

impl Repr for Candidate {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            MatchedBuiltinCandidate => format!("MatchedBuiltinCandidate"),
            AmbiguousBuiltinCandidate => format!("AmbiguousBuiltinCandidate"),
            MatchedUnboxedClosureCandidate(c) => format!("MatchedUnboxedClosureCandidate({})", c),
            MatchedParamCandidate(ref r) => format!("MatchedParamCandidate({})",
                                                    r.repr(tcx)),
            AmbiguousParamCandidate => format!("AmbiguousParamCandidate"),
            Impl(ref i) => i.repr(tcx)
        }
    }
}

impl Repr for ImplCandidate {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            MatchedImplCandidate(ref d) => format!("MatchedImplCandidate({})",
                                                   d.repr(tcx)),
            AmbiguousImplCandidate(ref d) => format!("AmbiguousImplCandidate({})",
                                                     d.repr(tcx)),
        }
    }
}


// impl SelectionCache {
//     pub fn new() -> SelectionCache {
//         SelectionCache {
//             hashmap: RefCell::new(HashMap::new())
//         }
//     }
// }

// impl CacheKey {
//     pub fn new(trait_def_id: ast::DefId,
//                skol_obligation_self_ty: ty::t)
//                -> CacheKey
//     {
//         CacheKey {
//             trait_def_id: trait_def_id,
//             skol_obligation_self_ty: skol_obligation_self_ty
//         }
//     }
// }
