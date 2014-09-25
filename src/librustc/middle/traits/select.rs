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

use middle::mem_categorization::Typer;
use middle::subst::{Subst, Substs, VecPerParamSpace};
use middle::ty;
use middle::ty_fold::TypeFoldable;
use middle::typeck::check::regionmanip;
use middle::typeck::infer;
use middle::typeck::infer::{InferCtxt, TypeSkolemizer};
use std::cell::RefCell;
use std::collections::hashmap::HashMap;
use std::rc::Rc;
use syntax::ast;
use util::ppaux::Repr;

pub struct SelectionContext<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    param_env: &'cx ty::ParameterEnvironment,
    typer: &'cx Typer<'tcx>+'cx,
    skolemizer: TypeSkolemizer<'cx, 'tcx>,
}

// A stack that walks back up the stack frame.
struct ObligationStack<'prev> {
    obligation: &'prev Obligation,
    skol_obligation_self_ty: ty::t,
    previous: Option<&'prev ObligationStack<'prev>>
}

pub struct SelectionCache {
    hashmap: RefCell<HashMap<CacheKey, SelectionResult<Candidate>>>,
}

#[deriving(Hash,Eq,PartialEq)]
struct CacheKey {
    trait_def_id: ast::DefId,
    skol_obligation_self_ty: ty::t,
}

#[deriving(PartialEq,Eq)]
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
    MatchedUnboxedClosureCandidate(/* closure */ ast::DefId),
    ErrorCandidate,
}

#[deriving(Clone)]
enum ImplCandidate {
    MatchedImplCandidate(ast::DefId),
    AmbiguousImplCandidate(ast::DefId),
}

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    pub fn new(infcx: &'cx InferCtxt<'cx, 'tcx>,
               param_env: &'cx ty::ParameterEnvironment,
               typer: &'cx Typer<'tcx>)
               -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
            param_env: param_env,
            typer: typer,
            skolemizer: infcx.skolemizer(),
        }
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

    pub fn select(&mut self, obligation: &Obligation) -> SelectionResult<Selection> {
        /*!
         * Evaluates whether the obligation can be satisfied. Returns
         * an indication of whether the obligation can be satisfied
         * and, if so, by what means. Never affects surrounding typing
         * environment.
         */

        debug!("select({})", obligation.repr(self.tcx()));

        let stack = self.new_stack(obligation);
        match try!(self.candidate_from_obligation(&stack)) {
            None => Ok(None),
            Some(candidate) => self.confirm_candidate(obligation, candidate),
        }
    }

    pub fn select_inherent_impl(&mut self,
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

    pub fn evaluate_obligation(&mut self,
                               obligation: &Obligation)
                               -> EvaluationResult
    {
        /*!
         * Evaluates whether the obligation `obligation` can be
         * satisfied (by any means).
         */

        debug!("evaluate_obligation({})",
               obligation.repr(self.tcx()));

        let stack = self.new_stack(obligation);
        match self.candidate_from_obligation(&stack) {
            Ok(Some(c)) => c.to_evaluation_result(),
            Ok(None) => EvaluatedToAmbiguity,
            Err(_) => EvaluatedToUnmatch,
        }
    }

    fn evaluate_builtin_bound_recursively(&mut self,
                                          bound: ty::BuiltinBound,
                                          previous_stack: &ObligationStack,
                                          ty: ty::t)
                                          -> EvaluationResult
    {
        let obligation =
            util::obligation_for_builtin_bound(
                self.tcx(),
                previous_stack.obligation.cause,
                bound,
                previous_stack.obligation.recursion_depth + 1,
                ty);
        self.evaluate_obligation_recursively(previous_stack, &obligation)
    }

    fn evaluate_obligation_recursively(&mut self,
                                       previous_stack: &ObligationStack,
                                       obligation: &Obligation)
                                       -> EvaluationResult
    {
        debug!("evaluate_obligation_recursively({})",
               obligation.repr(self.tcx()));

        // If there is any previous entry on the stack that precisely
        // matches this obligation, then we can assume that the
        // obligation is satisfied for now (still all other conditions
        // must be met of course). One obvious case this comes up is
        // marker traits like `Send`. Think of a a linked list:
        //
        //    struct List<T> { data: T, next: Option<Box<List<T>>> {
        //
        // `Box<List<T>>` will be `Send` if `T` is `Send` and
        // `Option<Box<List<T>>>` is `Send`, and in turn
        // `Option<Box<List<T>>>` is `Send` if `Box<List<T>>` is
        // `Send`.
        if
            previous_stack.iter()
            .filter(|e| e.obligation.trait_ref.def_id == obligation.trait_ref.def_id)
            .find(|e| self.match_self_types(obligation.cause,
                                            e.skol_obligation_self_ty,
                                            obligation.self_ty()) == Matched(()))
            .is_some()
        {
            return EvaluatedToMatch;
        }

        let stack = self.push_stack(previous_stack, obligation);
        match self.candidate_from_obligation(&stack) {
            Ok(Some(c)) => c.to_evaluation_result(),
            Ok(None) => EvaluatedToAmbiguity,
            Err(_) => EvaluatedToUnmatch,
        }
    }

    pub fn evaluate_impl(&mut self,
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

    fn candidate_from_obligation(&mut self,
                                 stack: &ObligationStack)
                                 -> SelectionResult<Candidate>
    {
        debug!("candidate_from_obligation({})",
               stack.repr(self.tcx()));

        // First, check the cache.
        match self.check_candidate_cache(stack.obligation, stack.skol_obligation_self_ty) {
            Some(c) => {
                debug!("check_candidate_cache(obligation={}, skol_obligation_self_ty={}, \
                       candidate={})",
                       stack.obligation.trait_ref.def_id,
                       stack.skol_obligation_self_ty.repr(self.tcx()),
                       c.repr(self.tcx()));
                return c;
            }
            None => { }
        }

        // If no match, compute result and insert into cache.
        let result = self.pick_candidate(stack);
        self.insert_candidate_cache(stack.obligation,
                                    stack.skol_obligation_self_ty,
                                    result.clone());
        result
    }

    fn pick_candidate(&mut self,
                      stack: &ObligationStack)
                      -> SelectionResult<Candidate>
    {
        if ty::type_is_error(stack.skol_obligation_self_ty) {
            return Ok(Some(ErrorCandidate));
        }

        let mut candidates = try!(self.assemble_candidates(stack));

        debug!("assembled {} candidates for {}",
               candidates.len(), stack.repr(self.tcx()));

        // Examine candidates to determine outcome. Ideally we will
        // have exactly one candidate that is definitively applicable.

        if candidates.len() == 0 {
            // Annoying edge case: if there are no impls, then there
            // is no way that this trait reference is implemented,
            // *unless* it contains unbound variables. In that case,
            // it is possible that one of those unbound variables will
            // be bound to a new type from some other crate which will
            // also contain impls.
            return if !self.contains_skolemized_types(stack.skol_obligation_self_ty) {
                debug!("0 matches, unimpl");
                Err(Unimplemented)
            } else {
                debug!("0 matches, ambig");
                Ok(None)
            }
        } else if candidates.len() > 1 {
            // Ambiguity. Possibly we should report back more
            // information on the potential candidates so we can give
            // a better error message.
            debug!("multiple matches, ambig");
            Ok(None)
        } else {
            let candidate = candidates.pop().unwrap();
            Ok(Some(candidate))
        }
    }

    fn pick_candidate_cache(&self,
                            _obligation: &Obligation,
                            skol_obligation_self_ty: ty::t)
                            -> &SelectionCache
    {
        if
            ty::type_has_self(skol_obligation_self_ty) ||
            ty::type_has_params(skol_obligation_self_ty)
        {
            &self.param_env.selection_cache
        } else {
            &self.tcx().selection_cache
        }
    }

    fn check_candidate_cache(&mut self,
                             obligation: &Obligation,
                             skol_obligation_self_ty: ty::t)
                             -> Option<SelectionResult<Candidate>>
    {
        let cache = self.pick_candidate_cache(obligation, skol_obligation_self_ty);
        let cache_key = CacheKey::new(obligation.trait_ref.def_id,
                                      skol_obligation_self_ty);
        let hashmap = cache.hashmap.borrow();
        hashmap.find(&cache_key).map(|c| (*c).clone())
    }

    fn insert_candidate_cache(&mut self,
                              obligation: &Obligation,
                              skol_obligation_self_ty: ty::t,
                              candidate: SelectionResult<Candidate>)
    {
        debug!("insert_candidate_cache(obligation={}, skol_obligation_self_ty={}, candidate={})",
               obligation.trait_ref.def_id,
               skol_obligation_self_ty.repr(self.tcx()),
               candidate.repr(self.tcx()));

        let cache = self.pick_candidate_cache(obligation, skol_obligation_self_ty);
        let cache_key = CacheKey::new(obligation.trait_ref.def_id,
                                      skol_obligation_self_ty);
        let mut hashmap = cache.hashmap.borrow_mut();
        hashmap.insert(cache_key, candidate);
    }

    fn assemble_candidates(&mut self,
                           stack: &ObligationStack)
                           -> Result<Vec<Candidate>, SelectionError>
    {
        // Check for overflow.

        let ObligationStack { obligation, skol_obligation_self_ty, .. } = *stack;

        let recursion_limit = self.infcx.tcx.sess.recursion_limit.get();
        if obligation.recursion_depth >= recursion_limit {
            debug!("{} --> overflow", stack.obligation.repr(self.tcx()));
            return Err(Overflow);
        }

        let mut candidates = Vec::new();

        // Other bounds. Consider both in-scope bounds from fn decl
        // and applicable impls. There is a certain set of precedence rules here.

        // Where clauses have highest precedence.
        try!(self.assemble_candidates_from_caller_bounds(
            obligation,
            skol_obligation_self_ty,
            &mut candidates));

        // In the special case of builtin bounds, consider the "compiler-supplied" impls.
        if candidates.len() == 0 {
            match self.tcx().lang_items.to_builtin_kind(obligation.trait_ref.def_id) {
                Some(bound) => {
                    try!(self.assemble_builtin_bound_candidates(bound, stack, &mut candidates));
                }

                None => { }
            }
        }

        // In the special case of fn traits and synthesized unboxed
        // closure types, consider the compiler-supplied impls. Note
        // that this is exclusive with the builtin bound case above.
        if candidates.len() == 0 {
            try!(self.assemble_unboxed_candidates(
                obligation,
                skol_obligation_self_ty,
                &mut candidates));
        }

        // Finally, consider the actual impls found in the program.
        if candidates.len() == 0 {
            try!(self.assemble_candidates_from_impls(
                obligation,
                skol_obligation_self_ty,
                &mut candidates));
        }

        Ok(candidates)
    }

    fn assemble_candidates_from_caller_bounds(&mut self,
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
         */

        debug!("assemble_candidates_from_caller_bounds({}, {})",
               obligation.repr(self.tcx()),
               skol_obligation_self_ty.repr(self.tcx()));

        for caller_obligation in self.param_env.caller_obligations.iter() {
            // Skip over obligations that don't apply to
            // `self_ty`.
            let caller_bound = &caller_obligation.trait_ref;
            let caller_self_ty = caller_bound.substs.self_ty().unwrap();
            debug!("caller_obligation={}, caller_self_ty={}",
                   caller_obligation.repr(self.tcx()),
                   self.infcx.ty_to_string(caller_self_ty));
            match self.match_self_types(obligation.cause,
                                        caller_self_ty,
                                        skol_obligation_self_ty) {
                AmbiguousMatch => {
                    debug!("-> AmbiguousMatch");
                    candidates.push(AmbiguousParamCandidate);
                    return Ok(());
                }
                NoMatch => {
                    debug!("-> NoMatch");
                    continue;
                }
                Matched(()) => { }
            }

            // Search through the trait (and its supertraits) to
            // see if it matches the def-id we are looking for.
            let caller_bound = (*caller_bound).clone();
            for bound in util::transitive_bounds(self.tcx(), &[caller_bound]) {
                debug!("-> check bound={}", bound.repr(self.tcx()));
                if bound.def_id == obligation.trait_ref.def_id {
                    // If so, we're done!
                    debug!("-> MatchedParamCandidate({})", bound.repr(self.tcx()));
                    let vtable_param = VtableParamData { bound: bound };
                    candidates.push(MatchedParamCandidate(vtable_param));
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn assemble_unboxed_candidates(&mut self,
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
            let closure_kind = match self.typer.unboxed_closures().borrow().find(&closure_def_id) {
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

    fn assemble_candidates_from_impls(&mut self,
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

    fn candidate_from_impl(&mut self,
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
    // BUILTIN BOUNDS
    //
    // These cover the traits that are built-in to the language
    // itself.  This includes `Copy` and `Sized` for sure. For the
    // moment, it also includes `Send` / `Sync` and a few others, but
    // those will hopefully change to library-defined traits in the
    // future.

    fn assemble_builtin_bound_candidates(&mut self,
                                         bound: ty::BuiltinBound,
                                         stack: &ObligationStack,
                                         candidates: &mut Vec<Candidate>)
                                         -> Result<(),SelectionError>
    {
        // Copy -- owned, dtor, managed, marker, &mut -- only INTERIOR?
        // Sized -- str, [T], Trait -- but only INTERIOR
        // Send -- managed data, nonsend annot, borrowed data -- REACHABILITY
        // Sync -- non-sync marker trait -- REACHABILITY

        // Ideally, we'd only have to examine the immediate fields.
        // But think this through carefully I guess.

        enum WhenOk<'a> {
            Always,
            Unknown,
            Never,
            If(ty::t),
            IfAll(&'a [ty::t]),
            IfTrue(bool)
        }

        let ok = |this: &mut SelectionContext, w: WhenOk| {
            let r = match w {
                Always => EvaluatedToMatch,
                Unknown => EvaluatedToAmbiguity,
                Never => EvaluatedToUnmatch,
                IfTrue(true) => EvaluatedToMatch,
                IfTrue(false) => EvaluatedToUnmatch,
                If(ty) => this.evaluate_builtin_bound_recursively(bound, stack, ty),
                IfAll(tys) => {
                    let mut result = EvaluatedToMatch;
                    for &ty in tys.iter() {
                        match this.evaluate_builtin_bound_recursively(bound, stack, ty) {
                            EvaluatedToMatch => { }
                            EvaluatedToAmbiguity => {
                                result = EvaluatedToAmbiguity;
                            }
                            EvaluatedToUnmatch => {
                                result = EvaluatedToUnmatch;
                                break;
                            }
                        }
                    }
                    result
                }
            };

            match r {
                EvaluatedToMatch => Ok(candidates.push(MatchedBuiltinCandidate)),
                EvaluatedToAmbiguity => Ok(candidates.push(AmbiguousBuiltinCandidate)),
                EvaluatedToUnmatch => Err(Unimplemented)
            }
        };

        return match ty::get(stack.skol_obligation_self_ty).sty {
            ty::ty_uint(_) | ty::ty_int(_) | ty::ty_infer(ty::SkolemizedIntTy(_)) |
            ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_float(_) |
            ty::ty_bare_fn(_) | ty::ty_char => {
                // safe for everything
                ok(self, Always)
            }

            ty::ty_box(_) => {
                match bound {
                    ty::BoundSync |
                    ty::BoundSend |
                    ty::BoundCopy => {
                        // Managed data is not copyable, sendable, nor
                        // synchronized, regardless of referent.
                        ok(self, Never)
                    }

                    ty::BoundSized => {
                        // But it is sized, regardless of referent.
                        ok(self, Always)
                    }
                }
            }

            ty::ty_uniq(referent_ty) => {  // Box<T>
                match bound {
                    ty::BoundCopy => {
                        ok(self, Never)
                    }

                    ty::BoundSized => {
                        ok(self, Always)
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        ok(self, If(referent_ty))
                    }
                }
            }

            ty::ty_ptr(ty::mt { ty: referent_ty, .. }) => {     // *const T, *mut T
                match bound {
                    ty::BoundCopy |
                    ty::BoundSized => {
                        ok(self, Always)
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        ok(self, If(referent_ty))
                    }
                }
            }

            ty::ty_closure(ref c) => {
                match c.store {
                    ty::UniqTraitStore => {
                        // proc: Equivalent to `Box<FnOnce>`
                        match bound {
                            ty::BoundCopy => {
                                ok(self, Never)
                            }

                            ty::BoundSized => {
                                ok(self, Always)
                            }

                            ty::BoundSync |
                            ty::BoundSend => {
                                ok(self, IfTrue(c.bounds.builtin_bounds.contains_elem(bound)))
                            }
                        }
                    }
                    ty::RegionTraitStore(_, mutbl) => {
                        // ||: Equivalent to `&FnMut` or `&mut FnMut` or something like that.
                        match bound {
                            ty::BoundCopy => {
                                ok(self, match mutbl {
                                    ast::MutMutable => Never,  // &mut T is affine
                                    ast::MutImmutable => Always,  // &T is copyable
                                })
                            }

                            ty::BoundSized => {
                                ok(self, Always)
                            }

                            ty::BoundSync |
                            ty::BoundSend => {
                                ok(self, IfTrue(c.bounds.builtin_bounds.contains_elem(bound)))
                            }
                        }
                    }
                }
            }

            ty::ty_trait(box ty::TyTrait { bounds, .. }) => {
                match bound {
                    ty::BoundSized => {
                        ok(self, Never)
                    }
                    ty::BoundCopy | ty::BoundSync | ty::BoundSend => {
                        ok(self, IfTrue(bounds.builtin_bounds.contains_elem(bound)))
                    }
                }
            }

            ty::ty_rptr(_, ty::mt { ty: referent_ty, mutbl: mutbl }) => {
                // &mut T or &T
                match bound {
                    ty::BoundCopy => {
                        ok(self, match mutbl {
                            ast::MutMutable => Never,  // &mut T is affine and hence never `Copy`
                            ast::MutImmutable => Always,  // &T is copyable
                        })
                    }

                    ty::BoundSized => {
                        ok(self, Always)
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        // Note: technically, a region pointer is only
                        // sendable if it has lifetime
                        // `'static`. However, we don't take regions
                        // into account when doing trait matching:
                        // instead, when we decide that `T : Send`, we
                        // will register a separate constraint with
                        // the region inferencer that `T : 'static`
                        // holds as well (because the trait `Send`
                        // requires it). This will ensure that there
                        // is no borrowed data in `T` (or else report
                        // an inference error). The reason we do it
                        // this way is that we do not yet *know* what
                        // lifetime the borrowed reference has, since
                        // we haven't finished running inference -- in
                        // other words, there's a kind of
                        // chicken-and-egg problem.
                        ok(self, If(referent_ty))
                    }
                }
            }

            ty::ty_vec(element_ty, ref len) => {
                // [T, ..n] and [T]
                match bound {
                    ty::BoundCopy => {
                        match *len {
                            Some(_) => ok(self, If(element_ty)), // [T, ..n] is copy iff T is copy
                            None => ok(self, Never), // [T] is unsized and hence affine
                        }
                    }

                    ty::BoundSized => {
                        ok(self, IfTrue(len.is_some()))
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        ok(self, If(element_ty))
                    }
                }
            }

            ty::ty_str => {
                // Equivalent to [u8]
                match bound {
                    ty::BoundSync |
                    ty::BoundSend => {
                        ok(self, Always)
                    }

                    ty::BoundCopy |
                    ty::BoundSized => {
                        ok(self, Never)
                    }
                }
            }

            ty::ty_tup(ref tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                ok(self, IfAll(tys.as_slice()))
            }

            ty::ty_unboxed_closure(def_id, _) => {
                // FIXME -- This case is tricky. In the case of by-ref
                // closures particularly, we need the results of
                // inference to decide how to reflect the type of each
                // upvar (the upvar may have type `T`, but the runtime
                // type could be `&mut`, `&`, or just `T`). For now,
                // though, we'll do this unsoundly and assume that all
                // captures are by value. Really what we ought to do
                // is reserve judgement and then intertwine this
                // analysis with closure inference.
                //
                // FIXME -- this is wrong with respect to
                // skolemization. We want to skolemize the types of
                // the variables, but to do THAT we need the ability
                // to "start" the skolemization numbering from a
                // higher point. Perhaps this just means creating a
                // single skolemizer and then using it again here?
                assert_eq!(def_id.krate, ast::LOCAL_CRATE);
                match self.tcx().freevars.borrow().find(&def_id.node) {
                    None => {
                        // No upvars.
                        ok(self, Always)
                    }

                    Some(freevars) => {
                        let tys: Vec<ty::t> =
                            freevars
                            .iter()
                            .map(|freevar| {
                                let freevar_def_id = freevar.def.def_id();
                                let freevar_ty = self.typer.node_ty(freevar_def_id.node)
                                    .unwrap_or(ty::mk_err());
                                freevar_ty.fold_with(&mut self.skolemizer)
                            })
                            .collect();
                        ok(self, IfAll(tys.as_slice()))
                    }
                }
            }

            ty::ty_struct(def_id, ref substs) => {
                let types: Vec<ty::t> =
                    ty::struct_fields(self.tcx(), def_id, substs)
                    .iter()
                    .map(|f| f.mt.ty)
                    .collect();
                nominal(self, bound, def_id, types, ok)
            }

            ty::ty_enum(def_id, ref substs) => {
                let types: Vec<ty::t> =
                    ty::substd_enum_variants(self.tcx(), def_id, substs)
                    .iter()
                    .flat_map(|variant| variant.args.iter())
                    .map(|&ty| ty)
                    .collect();
                nominal(self, bound, def_id, types, ok)
            }

            ty::ty_param(_) => {
                // Note: A type parameter is only considered to meet a
                // particular bound if there is a where clause telling
                // us that it does, and that case is handled by
                // `assemble_candidates_from_caller_bounds()`.
                ok(self, Never)
            }

            ty::ty_infer(ty::SkolemizedTy(_)) => {
                // Skolemized types represent unbound type
                // variables. They might or might not have applicable
                // impls and so forth, depending on what those type
                // variables wind up being bound to.
                ok(self, Unknown)
            }

            ty::ty_open(_) |
            ty::ty_infer(ty::TyVar(_)) |
            ty::ty_infer(ty::IntVar(_)) |
            ty::ty_infer(ty::FloatVar(_)) |
            ty::ty_err => {
                self.tcx().sess.span_bug(
                    stack.obligation.cause.span,
                    format!(
                        "asked to compute contents of unexpected type: {}",
                        stack.skol_obligation_self_ty.repr(self.tcx())).as_slice());
            }
        };

        fn nominal(this: &mut SelectionContext,
                   bound: ty::BuiltinBound,
                   def_id: ast::DefId,
                   types: Vec<ty::t>,
                   ok: |&mut SelectionContext, WhenOk| -> Result<(),SelectionError>)
                   -> Result<(),SelectionError>
        {
            // First check for markers and other nonsense.
            let tcx = this.tcx();
            match bound {
                ty::BoundSend => {
                    if
                        Some(def_id) == tcx.lang_items.no_send_bound() ||
                        Some(def_id) == tcx.lang_items.managed_bound()
                    {
                        return ok(this, Never);
                    }
                }

                ty::BoundCopy => {
                    if
                        Some(def_id) == tcx.lang_items.no_copy_bound() ||
                        Some(def_id) == tcx.lang_items.managed_bound() ||
                        ty::has_dtor(tcx, def_id)
                    {
                        return ok(this, Never);
                    }
                }

                ty::BoundSync => {
                    if
                        Some(def_id) == tcx.lang_items.no_sync_bound() ||
                        Some(def_id) == tcx.lang_items.managed_bound()
                    {
                        return ok(this, Never);
                    } else if
                        Some(def_id) == tcx.lang_items.unsafe_type()
                    {
                        // FIXME(#13231) -- we currently consider `UnsafeCell<T>`
                        // to always be sync. This is allow for types like `Queue`
                        // and `Mutex`, where `Queue<T> : Sync` is `T : Send`.
                        return ok(this, Always);
                    }
                }

                ty::BoundSized => { }
            }

            ok(this, IfAll(types.as_slice()))
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // CONFIRMATION
    //
    // Confirmation unifies the output type parameters of the trait
    // with the values found in the obligation, possibly yielding a
    // type error.  See `doc.rs` for more details.

    fn confirm_candidate(&mut self,
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

            ErrorCandidate |
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

    fn confirm_param_candidate(&mut self,
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

    fn confirm_impl_candidate(&mut self,
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

    fn confirm_inherent_impl_candidate(&mut self,
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

    fn confirm_unboxed_closure_candidate(&mut self,
                                         obligation: &Obligation,
                                         closure_def_id: ast::DefId)
                                         -> Result<(),SelectionError>
    {
        debug!("confirm_unboxed_closure_candidate({},{})",
               obligation.repr(self.tcx()),
               closure_def_id.repr(self.tcx()));

        let closure_type = match self.typer.unboxed_closures().borrow().find(&closure_def_id) {
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

    fn match_impl_self_types(&mut self,
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

    fn match_self_types(&mut self,
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

    fn confirm_impl_vtable(&mut self,
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

    fn confirm(&mut self,
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

    fn new_stack<'o>(&mut self, obligation: &'o Obligation) -> ObligationStack<'o> {
        let skol_obligation_self_ty =
            obligation.self_ty().fold_with(&mut self.skolemizer);

        ObligationStack {
            obligation: obligation,
            skol_obligation_self_ty: skol_obligation_self_ty,
            previous: None
        }
    }

    fn push_stack<'o>(&self,
                      previous_stack: &'o ObligationStack<'o>,
                      obligation: &'o Obligation)
                      -> ObligationStack<'o>
    {
        // No need to skolemize obligation.self_ty, because we
        // guarantee the self-type for all recursive obligations are
        // already skolemized.
        ObligationStack {
            obligation: obligation,
            skol_obligation_self_ty: obligation.self_ty(),
            previous: Some(previous_stack)
        }
    }

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

    fn contains_skolemized_types(&self,
                                 ty: ty::t)
                                 -> bool
    {
        /*!
         * True if the type contains skolemized variables.
         */

        let mut found_skol = false;

        ty::walk_ty(ty, |t| {
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

            ErrorCandidate |
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
            ErrorCandidate => format!("ErrorCandidate"),
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

impl SelectionCache {
    pub fn new() -> SelectionCache {
        SelectionCache {
            hashmap: RefCell::new(HashMap::new())
        }
    }
}

impl<'o> ObligationStack<'o> {
    fn iter(&self) -> Option<&ObligationStack> {
        Some(self)
    }
}

impl<'o> Iterator<&'o ObligationStack<'o>> for Option<&'o ObligationStack<'o>> {
    fn next(&mut self) -> Option<&'o ObligationStack<'o>> {
        match *self {
            Some(o) => {
                *self = o.previous;
                Some(o)
            }
            None => {
                None
            }
        }
    }
}

impl<'o> Repr for ObligationStack<'o> {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("ObligationStack({}, {})",
                self.obligation.repr(tcx),
                self.skol_obligation_self_ty.repr(tcx))
    }
}

impl CacheKey {
    pub fn new(trait_def_id: ast::DefId,
               skol_obligation_self_ty: ty::t)
               -> CacheKey
    {
        CacheKey {
            trait_def_id: trait_def_id,
            skol_obligation_self_ty: skol_obligation_self_ty
        }
    }
}
