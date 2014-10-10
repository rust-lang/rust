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
#![allow(dead_code)] // FIXME -- just temporarily

use super::{ErrorReported};
use super::{Obligation, ObligationCause};
use super::{SelectionError, Unimplemented, Overflow,
            OutputTypeParameterMismatch};
use super::{Selection};
use super::{SelectionResult};
use super::{VtableBuiltin, VtableImpl, VtableParam, VtableUnboxedClosure};
use super::{VtableImplData, VtableParamData, VtableBuiltinData};
use super::{util};

use middle::mem_categorization::Typer;
use middle::subst::{Subst, Substs, VecPerParamSpace};
use middle::ty;
use middle::typeck::check::regionmanip;
use middle::typeck::infer;
use middle::typeck::infer::{InferCtxt, TypeSkolemizer};
use middle::ty_fold::TypeFoldable;
use std::cell::RefCell;
use std::collections::hashmap::HashMap;
use std::rc::Rc;
use std::result;
use syntax::ast;
use util::ppaux::Repr;

pub struct SelectionContext<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    param_env: &'cx ty::ParameterEnvironment,
    typer: &'cx Typer<'tcx>+'cx,

    /// Skolemizer used specifically for skolemizing entries on the
    /// obligation stack. This ensures that all entries on the stack
    /// at one time will have the same set of skolemized entries,
    /// which is important for checking for trait bounds that
    /// recursively require themselves.
    skolemizer: TypeSkolemizer<'cx, 'tcx>,
}

// A stack that walks back up the stack frame.
struct ObligationStack<'prev> {
    obligation: &'prev Obligation,

    /// Trait ref from `obligation` but skolemized with the
    /// selection-context's skolemizer. Used to check for recursion.
    skol_trait_ref: Rc<ty::TraitRef>,

    previous: Option<&'prev ObligationStack<'prev>>
}

pub struct SelectionCache {
    hashmap: RefCell<HashMap<Rc<ty::TraitRef>, SelectionResult<Candidate>>>,
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
#[deriving(PartialEq,Eq,Show,Clone)]
enum Candidate {
    BuiltinCandidate(ty::BuiltinBound),
    ParamCandidate(VtableParamData),
    ImplCandidate(ast::DefId),
    UnboxedClosureCandidate(/* closure */ ast::DefId),
    ErrorCandidate,
}

struct CandidateSet {
    vec: Vec<Candidate>,
    ambiguous: bool
}

enum BuiltinBoundConditions {
    If(Vec<ty::t>),
    ParameterBuiltin,
    AmbiguousBuiltin
}

#[deriving(Show)]
enum EvaluationResult {
    EvaluatedToOk,
    EvaluatedToErr,
    EvaluatedToAmbig,
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

        let stack = self.push_stack(None, obligation);
        match try!(self.candidate_from_obligation(&stack)) {
            None => Ok(None),
            Some(candidate) => Ok(Some(try!(self.confirm_candidate(obligation, candidate)))),
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

        match self.match_inherent_impl(impl_def_id,
                                       obligation_cause,
                                       obligation_self_ty) {
            Ok(substs) => {
                let vtable_impl = self.vtable_impl(impl_def_id, substs, obligation_cause, 0);
                Ok(Some(vtable_impl))
            }
            Err(()) => {
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
    //
    // The result is "true" if the obliation *may* hold and "false" if
    // we can be sure it does not.

    pub fn evaluate_obligation(&mut self,
                               obligation: &Obligation)
                               -> bool
    {
        /*!
         * Evaluates whether the obligation `obligation` can be
         * satisfied (by any means).
         */

        debug!("evaluate_obligation({})",
               obligation.repr(self.tcx()));

        let stack = self.push_stack(None, obligation);
        self.evaluate_stack(&stack).may_apply()
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

        match obligation {
            Ok(obligation) => {
                self.evaluate_obligation_recursively(Some(previous_stack), &obligation)
            }
            Err(ErrorReported) => {
                EvaluatedToOk
            }
        }
    }

    fn evaluate_obligation_recursively(&mut self,
                                       previous_stack: Option<&ObligationStack>,
                                       obligation: &Obligation)
                                       -> EvaluationResult
    {
        debug!("evaluate_obligation_recursively({})",
               obligation.repr(self.tcx()));

        let stack = self.push_stack(previous_stack.map(|x| x), obligation);
        let result = self.evaluate_stack(&stack);
        debug!("result: {}", result);
        result
    }

    fn evaluate_stack(&mut self,
                      stack: &ObligationStack)
                      -> EvaluationResult
    {
        // Whenever any of the types are unbound, there can always be
        // an impl.  Even if there are no impls in this crate, perhaps
        // the type would be unified with something from another crate
        // that does provide an impl.
        let input_types = &stack.skol_trait_ref.substs.types;
        if input_types.iter().any(|&t| ty::type_is_skolemized(t)) {
            debug!("evaluate_stack({}) --> unbound argument, must be ambiguous",
                   stack.skol_trait_ref.repr(self.tcx()));
            return EvaluatedToAmbig;
        }

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
        //
        // Note that we do this comparison using the `skol_trait_ref`
        // fields. Because these have all been skolemized using
        // `self.skolemizer`, we can be sure that (a) this will not
        // affect the inferencer state and (b) that if we see two
        // skolemized types with the same index, they refer to the
        // same unbound type variable.
        if
            stack.iter()
            .skip(1) // skip top-most frame
            .any(|prev| stack.skol_trait_ref == prev.skol_trait_ref)
        {
            debug!("evaluate_stack({}) --> recursive",
                   stack.skol_trait_ref.repr(self.tcx()));
            return EvaluatedToOk;
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.winnow_candidate(stack, &c),
            Ok(None) => EvaluatedToAmbig,
            Err(_) => EvaluatedToErr,
        }
    }

    pub fn evaluate_impl(&mut self,
                         impl_def_id: ast::DefId,
                         obligation: &Obligation)
                         -> bool
    {
        /*!
         * Evaluates whether the impl with id `impl_def_id` could be
         * applied to the self type `obligation_self_ty`. This can be
         * used either for trait or inherent impls.
         */

        debug!("evaluate_impl(impl_def_id={}, obligation={})",
               impl_def_id.repr(self.tcx()),
               obligation.repr(self.tcx()));

        self.infcx.probe(|| {
            match self.match_impl(impl_def_id, obligation) {
                Ok(substs) => {
                    let vtable_impl = self.vtable_impl(impl_def_id, substs, obligation.cause, 0);
                    self.winnow_selection(None, VtableImpl(vtable_impl)).may_apply()
                }
                Err(()) => {
                    false
                }
            }
        })
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
        // Watch out for overflow. This intentionally bypasses (and does
        // not update) the cache.
        let recursion_limit = self.infcx.tcx.sess.recursion_limit.get();
        if stack.obligation.recursion_depth >= recursion_limit {
            debug!("{} --> overflow (limit={})",
                   stack.obligation.repr(self.tcx()),
                   recursion_limit);
            return Err(Overflow)
        }

        // Check the cache. Note that we skolemize the trait-ref
        // separately rather than using `stack.skol_trait_ref` -- this
        // is because we want the unbound variables to be replaced
        // with fresh skolemized types starting from index 0.
        let cache_skol_trait_ref =
            self.infcx.skolemize(stack.obligation.trait_ref.clone());
        debug!("candidate_from_obligation(cache_skol_trait_ref={}, obligation={})",
               cache_skol_trait_ref.repr(self.tcx()),
               stack.repr(self.tcx()));

        match self.check_candidate_cache(cache_skol_trait_ref.clone()) {
            Some(c) => {
                debug!("CACHE HIT: cache_skol_trait_ref={}, candidate={}",
                       cache_skol_trait_ref.repr(self.tcx()),
                       c.repr(self.tcx()));
                return c;
            }
            None => { }
        }

        // If no match, compute result and insert into cache.
        let candidate = self.candidate_from_obligation_no_cache(stack);
        debug!("CACHE MISS: cache_skol_trait_ref={}, candidate={}",
               cache_skol_trait_ref.repr(self.tcx()), candidate.repr(self.tcx()));
        self.insert_candidate_cache(cache_skol_trait_ref, candidate.clone());
        candidate
    }

    fn candidate_from_obligation_no_cache(&mut self,
                                          stack: &ObligationStack)
                                          -> SelectionResult<Candidate>
    {
        if ty::type_is_error(stack.obligation.self_ty()) {
            return Ok(Some(ErrorCandidate));
        }

        let candidate_set = try!(self.assemble_candidates(stack));

        if candidate_set.ambiguous {
            debug!("candidate set contains ambig");
            return Ok(None);
        }

        let mut candidates = candidate_set.vec;

        debug!("assembled {} candidates for {}",
               candidates.len(), stack.repr(self.tcx()));

        // At this point, we know that each of the entries in the
        // candidate set is *individually* applicable. Now we have to
        // figure out if they contain mutual incompatibilities. This
        // frequently arises if we have an unconstrained input type --
        // for example, we are looking for $0:Eq where $0 is some
        // unconstrained type variable. In that case, we'll get a
        // candidate which assumes $0 == int, one that assumes $0 ==
        // uint, etc. This spells an ambiguity.

        // If there is more than one candidate, first winnow them down
        // by considering extra conditions (nested obligations and so
        // forth). We don't winnow if there is exactly one
        // candidate. This is a relatively minor distinction but it
        // can lead to better inference and error-reporting. An
        // example would be if there was an impl:
        //
        //     impl<T:Clone> Vec<T> { fn push_clone(...) { ... } }
        //
        // and we were to see some code `foo.push_clone()` where `boo`
        // is a `Vec<Bar>` and `Bar` does not implement `Clone`.  If
        // we were to winnow, we'd wind up with zero candidates.
        // Instead, we select the right impl now but report `Bar does
        // not implement Clone`.
        if candidates.len() > 1 {
            candidates.retain(|c| self.winnow_candidate(stack, c).may_apply())
        }

        // If there are STILL multiple candidate, we can further reduce
        // the list by dropping duplicates.
        if candidates.len() > 1 {
            let mut i = 0;
            while i < candidates.len() {
                let is_dup =
                    range(0, candidates.len())
                    .filter(|&j| i != j)
                    .any(|j| self.candidate_should_be_dropped_in_favor_of(stack,
                                                                          &candidates[i],
                                                                          &candidates[j]));
                if is_dup {
                    debug!("Dropping candidate #{}/#{}: {}",
                           i, candidates.len(), candidates[i].repr(self.tcx()));
                    candidates.swap_remove(i);
                } else {
                    debug!("Retaining candidate #{}/#{}",
                           i, candidates.len());
                    i += 1;
                }
            }
        }

        // If there are *STILL* multiple candidates, give up and
        // report ambiguiuty.
        if candidates.len() > 1 {
            debug!("multiple matches, ambig");
            return Ok(None);
        }

        // If there are *NO* candidates, that might mean either that
        // there is no impl or just that we can't know anything for
        // sure.
        if candidates.len() == 0 {
            // Annoying edge case: if there are no impls, then there
            // is no way that this trait reference is implemented,
            // *unless* it contains unbound variables. In that case,
            // it is possible that one of those unbound variables will
            // be bound to a new type from some other crate which will
            // also contain impls.
            let skol_obligation_self_ty = self.infcx.skolemize(stack.obligation.self_ty());
            return if !self.contains_skolemized_types(skol_obligation_self_ty) {
                debug!("0 matches, unimpl");
                Err(Unimplemented)
            } else {
                debug!("0 matches, ambig");
                Ok(None)
            };
        }

        // Just one candidate left.
        let candidate = candidates.pop().unwrap();
        Ok(Some(candidate))
    }

    fn pick_candidate_cache(&self,
                            cache_skol_trait_ref: &Rc<ty::TraitRef>)
                            -> &SelectionCache
    {
        // If the trait refers to any parameters in scope, then use
        // the cache of the param-environment. This is because the
        // result will depend on the where clauses that are in
        // scope. Otherwise, use the generic tcx cache, since the
        // result holds across all environments.
        if
            cache_skol_trait_ref.substs.types.iter().any(
                |&t| ty::type_has_self(t) || ty::type_has_params(t))
        {
            &self.param_env.selection_cache
        } else {
            &self.tcx().selection_cache
        }
    }

    fn check_candidate_cache(&mut self,
                             cache_skol_trait_ref: Rc<ty::TraitRef>)
                             -> Option<SelectionResult<Candidate>>
    {
        let cache = self.pick_candidate_cache(&cache_skol_trait_ref);
        let hashmap = cache.hashmap.borrow();
        hashmap.find(&cache_skol_trait_ref).map(|c| (*c).clone())
    }

    fn insert_candidate_cache(&mut self,
                              cache_skol_trait_ref: Rc<ty::TraitRef>,
                              candidate: SelectionResult<Candidate>)
    {
        let cache = self.pick_candidate_cache(&cache_skol_trait_ref);
        let mut hashmap = cache.hashmap.borrow_mut();
        hashmap.insert(cache_skol_trait_ref, candidate);
    }

    fn assemble_candidates(&mut self,
                           stack: &ObligationStack)
                           -> Result<CandidateSet, SelectionError>
    {
        // Check for overflow.

        let ObligationStack { obligation, .. } = *stack;

        let mut candidates = CandidateSet {
            vec: Vec::new(),
            ambiguous: false
        };

        // Other bounds. Consider both in-scope bounds from fn decl
        // and applicable impls. There is a certain set of precedence rules here.

        match self.tcx().lang_items.to_builtin_kind(obligation.trait_ref.def_id) {
            Some(bound) => {
                try!(self.assemble_builtin_bound_candidates(bound, stack, &mut candidates));
            }

            None => {
                // For the time being, we ignore user-defined impls for builtin-bounds.
                // (And unboxed candidates only apply to the Fn/FnMut/etc traits.)
                try!(self.assemble_unboxed_candidates(obligation, &mut candidates));
                try!(self.assemble_candidates_from_impls(obligation, &mut candidates));
            }
        }

        try!(self.assemble_candidates_from_caller_bounds(obligation, &mut candidates));
        Ok(candidates)
    }

    fn assemble_candidates_from_caller_bounds(&mut self,
                                              obligation: &Obligation,
                                              candidates: &mut CandidateSet)
                                              -> Result<(),SelectionError>
    {
        /*!
         * Given an obligation like `<SomeTrait for T>`, search the obligations
         * that the caller supplied to find out whether it is listed among
         * them.
         *
         * Never affects inference environment.
         */

        debug!("assemble_candidates_from_caller_bounds({})",
               obligation.repr(self.tcx()));

        let caller_trait_refs: Vec<Rc<ty::TraitRef>> =
            self.param_env.caller_obligations.iter()
            .map(|o| o.trait_ref.clone())
            .collect();

        let all_bounds =
            util::transitive_bounds(
                self.tcx(), caller_trait_refs.as_slice());

        let matching_bounds =
            all_bounds.filter(
                |bound| self.infcx.probe(
                    || self.match_trait_refs(obligation,
                                             (*bound).clone())).is_ok());

        let param_candidates =
            matching_bounds.map(
                |bound| ParamCandidate(VtableParamData { bound: bound }));

        candidates.vec.extend(param_candidates);

        Ok(())
    }

    fn assemble_unboxed_candidates(&mut self,
                                   obligation: &Obligation,
                                   candidates: &mut CandidateSet)
                                   -> Result<(),SelectionError>
    {
        /*!
         * Check for the artificial impl that the compiler will create
         * for an obligation like `X : FnMut<..>` where `X` is an
         * unboxed closure type.
         *
         * Note: the type parameters on an unboxed closure candidate
         * are modeled as *output* type parameters and hence do not
         * affect whether this trait is a match or not. They will be
         * unified during the confirmation step.
         */

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        let closure_def_id = match ty::get(self_ty).sty {
            ty::ty_unboxed_closure(id, _) => id,
            ty::ty_infer(ty::TyVar(_)) => {
                candidates.ambiguous = true;
                return Ok(());
            }
            _ => { return Ok(()); }
        };

        debug!("assemble_unboxed_candidates: self_ty={} obligation={}",
               self_ty.repr(self.tcx()),
               obligation.repr(self.tcx()));

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

            candidates.vec.push(UnboxedClosureCandidate(closure_def_id));
        }

        Ok(())
    }

    fn assemble_candidates_from_impls(&mut self,
                                      obligation: &Obligation,
                                      candidates: &mut CandidateSet)
                                      -> Result<(), SelectionError>
    {
        /*!
         * Search for impls that might apply to `obligation`.
         */

        let all_impls = self.all_impls(obligation.trait_ref.def_id);
        for &impl_def_id in all_impls.iter() {
            self.infcx.probe(|| {
                match self.match_impl(impl_def_id, obligation) {
                    Ok(_) => {
                        candidates.vec.push(ImplCandidate(impl_def_id));
                    }
                    Err(()) => { }
                }
            });
        }
        Ok(())
    }

    ///////////////////////////////////////////////////////////////////////////
    // WINNOW
    //
    // Winnowing is the process of attempting to resolve ambiguity by
    // probing further. During the winnowing process, we unify all
    // type variables (ignoring skolemization) and then we also
    // attempt to evaluate recursive bounds to see if they are
    // satisfied.

    fn winnow_candidate(&mut self,
                        stack: &ObligationStack,
                        candidate: &Candidate)
                        -> EvaluationResult
    {
        /*!
         * Further evaluate `candidate` to decide whether all type parameters match
         * and whether nested obligations are met. Returns true if `candidate` remains
         * viable after this further scrutiny.
         */

        debug!("winnow_candidate: candidate={}", candidate.repr(self.tcx()));
        self.infcx.probe(|| {
            let candidate = (*candidate).clone();
            match self.confirm_candidate(stack.obligation, candidate) {
                Ok(selection) => self.winnow_selection(Some(stack), selection),
                Err(_) => EvaluatedToErr,
            }
        })
    }

    fn winnow_selection(&mut self,
                        stack: Option<&ObligationStack>,
                        selection: Selection)
                        -> EvaluationResult
    {
        let mut result = EvaluatedToOk;
        for obligation in selection.iter_nested() {
            match self.evaluate_obligation_recursively(stack, obligation) {
                EvaluatedToErr => { return EvaluatedToErr; }
                EvaluatedToAmbig => { result = EvaluatedToAmbig; }
                EvaluatedToOk => { }
            }
        }
        result
    }

    fn candidate_should_be_dropped_in_favor_of(&mut self,
                                               stack: &ObligationStack,
                                               candidate_i: &Candidate,
                                               candidate_j: &Candidate)
                                               -> bool
    {
        /*!
         * Returns true if `candidate_i` should be dropped in favor of `candidate_j`.
         * This is generally true if either:
         * - candidate i and candidate j are equivalent; or,
         * - candidate i is a where clause bound and candidate j is a concrete impl,
         *   and the concrete impl is applicable to the types in the where clause bound.
         *
         * The last case basically occurs with blanket impls like
         * `impl<T> Foo for T`.  In that case, a bound like `T:Foo` is
         * kind of an "false" ambiguity -- both are applicable to any
         * type, but in fact coherence requires that the bound will
         * always be resolved to the impl anyway.
         */

        match (candidate_i, candidate_j) {
            (&ParamCandidate(ref vt), &ImplCandidate(impl_def_id)) => {
                debug!("Considering whether to drop param {} in favor of impl {}",
                       candidate_i.repr(self.tcx()),
                       candidate_j.repr(self.tcx()));

                self.infcx.probe(|| {
                    let impl_substs =
                        self.rematch_impl(impl_def_id, stack.obligation);
                    let impl_trait_ref =
                        ty::impl_trait_ref(self.tcx(), impl_def_id).unwrap();
                    let impl_trait_ref =
                        impl_trait_ref.subst(self.tcx(), &impl_substs);
                    let origin =
                        infer::RelateOutputImplTypes(stack.obligation.cause.span);
                    self.infcx
                        .sub_trait_refs(false, origin,
                                        impl_trait_ref, vt.bound.clone())
                        .is_ok()
                })
            }
            _ => {
                *candidate_i == *candidate_j
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
                                         candidates: &mut CandidateSet)
                                         -> Result<(),SelectionError>
    {
        // FIXME -- To be more like a normal impl, we should just
        // ignore the nested cases here, and instead generate nested
        // obligations in `confirm_candidate`. However, this doesn't
        // work because we require handling the recursive cases to
        // avoid infinite cycles (that is, with recursive types,
        // sometimes `Foo : Copy` only holds if `Foo : Copy`).

        match self.builtin_bound(bound, stack.obligation.self_ty()) {
            Ok(If(nested)) => {
                debug!("builtin_bound: bound={} nested={}",
                       bound.repr(self.tcx()),
                       nested.repr(self.tcx()));
                let data = self.vtable_builtin_data(stack.obligation, bound, nested);
                match self.winnow_selection(Some(stack), VtableBuiltin(data)) {
                    EvaluatedToOk => { Ok(candidates.vec.push(BuiltinCandidate(bound))) }
                    EvaluatedToAmbig => { Ok(candidates.ambiguous = true) }
                    EvaluatedToErr => { Err(Unimplemented) }
                }
            }
            Ok(ParameterBuiltin) => { Ok(()) }
            Ok(AmbiguousBuiltin) => { Ok(candidates.ambiguous = true) }
            Err(e) => { Err(e) }
        }
    }

    fn builtin_bound(&mut self,
                     bound: ty::BuiltinBound,
                     self_ty: ty::t)
                     -> Result<BuiltinBoundConditions,SelectionError>
    {
        let self_ty = self.infcx.shallow_resolve(self_ty);
        return match ty::get(self_ty).sty {
            ty::ty_infer(ty::IntVar(_)) |
            ty::ty_infer(ty::FloatVar(_)) |
            ty::ty_uint(_) |
            ty::ty_int(_) |
            ty::ty_nil |
            ty::ty_bot |
            ty::ty_bool |
            ty::ty_float(_) |
            ty::ty_bare_fn(_) |
            ty::ty_char => {
                // safe for everything
                Ok(If(Vec::new()))
            }

            ty::ty_uniq(referent_ty) => {  // Box<T>
                match bound {
                    ty::BoundCopy => {
                        Err(Unimplemented)
                    }

                    ty::BoundSized => {
                        Ok(If(Vec::new()))
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        Ok(If(vec![referent_ty]))
                    }
                }
            }

            ty::ty_ptr(ty::mt { ty: referent_ty, .. }) => {     // *const T, *mut T
                match bound {
                    ty::BoundCopy |
                    ty::BoundSized => {
                        Ok(If(Vec::new()))
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        Ok(If(vec![referent_ty]))
                    }
                }
            }

            ty::ty_closure(ref c) => {
                match c.store {
                    ty::UniqTraitStore => {
                        // proc: Equivalent to `Box<FnOnce>`
                        match bound {
                            ty::BoundCopy => {
                                Err(Unimplemented)
                            }

                            ty::BoundSized => {
                                Ok(If(Vec::new()))
                            }

                            ty::BoundSync |
                            ty::BoundSend => {
                                if c.bounds.builtin_bounds.contains_elem(bound) {
                                    Ok(If(Vec::new()))
                                } else {
                                    Err(Unimplemented)
                                }
                            }
                        }
                    }
                    ty::RegionTraitStore(_, mutbl) => {
                        // ||: Equivalent to `&FnMut` or `&mut FnMut` or something like that.
                        match bound {
                            ty::BoundCopy => {
                                match mutbl {
                                    ast::MutMutable => Err(Unimplemented),  // &mut T is affine
                                    ast::MutImmutable => Ok(If(Vec::new())),  // &T is copyable
                                }
                            }

                            ty::BoundSized => {
                                Ok(If(Vec::new()))
                            }

                            ty::BoundSync |
                            ty::BoundSend => {
                                if c.bounds.builtin_bounds.contains_elem(bound) {
                                    Ok(If(Vec::new()))
                                } else {
                                    Err(Unimplemented)
                                }
                            }
                        }
                    }
                }
            }

            ty::ty_trait(box ty::TyTrait { bounds, .. }) => {
                match bound {
                    ty::BoundSized => {
                        Err(Unimplemented)
                    }
                    ty::BoundCopy | ty::BoundSync | ty::BoundSend => {
                        if bounds.builtin_bounds.contains_elem(bound) {
                            Ok(If(Vec::new()))
                        } else {
                            Err(Unimplemented)
                        }
                    }
                }
            }

            ty::ty_rptr(_, ty::mt { ty: referent_ty, mutbl: mutbl }) => {
                // &mut T or &T
                match bound {
                    ty::BoundCopy => {
                        match mutbl {
                            // &mut T is affine and hence never `Copy`
                            ast::MutMutable => Err(Unimplemented),

                            // &T is always copyable
                            ast::MutImmutable => Ok(If(Vec::new())),
                        }
                    }

                    ty::BoundSized => {
                        Ok(If(Vec::new()))
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
                        Ok(If(vec![referent_ty]))
                    }
                }
            }

            ty::ty_vec(element_ty, ref len) => {
                // [T, ..n] and [T]
                match bound {
                    ty::BoundCopy => {
                        match *len {
                            Some(_) => Ok(If(vec![element_ty])), // [T, ..n] is copy iff T is copy
                            None => Err(Unimplemented), // [T] is unsized and hence affine
                        }
                    }

                    ty::BoundSized => {
                        if len.is_some() {
                            Ok(If(Vec::new()))
                        } else {
                            Err(Unimplemented)
                        }
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        Ok(If(vec![element_ty]))
                    }
                }
            }

            ty::ty_str => {
                // Equivalent to [u8]
                match bound {
                    ty::BoundSync |
                    ty::BoundSend => {
                        Ok(If(Vec::new()))
                    }

                    ty::BoundCopy |
                    ty::BoundSized => {
                        Err(Unimplemented)
                    }
                }
            }

            ty::ty_tup(ref tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                Ok(If(tys.to_owned()))
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
                assert_eq!(def_id.krate, ast::LOCAL_CRATE);
                match self.tcx().freevars.borrow().find(&def_id.node) {
                    None => {
                        // No upvars.
                        Ok(If(Vec::new()))
                    }

                    Some(freevars) => {
                        let tys: Vec<ty::t> =
                            freevars
                            .iter()
                            .map(|freevar| {
                                let freevar_def_id = freevar.def.def_id();
                                self.typer.node_ty(freevar_def_id.node)
                                    .unwrap_or(ty::mk_err())
                            })
                            .collect();
                        Ok(If(tys))
                    }
                }
            }

            ty::ty_struct(def_id, ref substs) => {
                let types: Vec<ty::t> =
                    ty::struct_fields(self.tcx(), def_id, substs)
                    .iter()
                    .map(|f| f.mt.ty)
                    .collect();
                nominal(self, bound, def_id, types)
            }

            ty::ty_enum(def_id, ref substs) => {
                let types: Vec<ty::t> =
                    ty::substd_enum_variants(self.tcx(), def_id, substs)
                    .iter()
                    .flat_map(|variant| variant.args.iter())
                    .map(|&ty| ty)
                    .collect();
                nominal(self, bound, def_id, types)
            }

            ty::ty_param(_) => {
                // Note: A type parameter is only considered to meet a
                // particular bound if there is a where clause telling
                // us that it does, and that case is handled by
                // `assemble_candidates_from_caller_bounds()`.
                Ok(ParameterBuiltin)
            }

            ty::ty_infer(ty::TyVar(_)) => {
                // Unbound type variable. Might or might not have
                // applicable impls and so forth, depending on what
                // those type variables wind up being bound to.
                Ok(AmbiguousBuiltin)
            }

            ty::ty_err => {
                Ok(If(Vec::new()))
            }

            ty::ty_open(_) |
            ty::ty_infer(ty::SkolemizedTy(_)) |
            ty::ty_infer(ty::SkolemizedIntTy(_)) => {
                self.tcx().sess.bug(
                    format!(
                        "asked to assemble builtin bounds of unexpected type: {}",
                        self_ty.repr(self.tcx())).as_slice());
            }
        };

        fn nominal(this: &mut SelectionContext,
                   bound: ty::BuiltinBound,
                   def_id: ast::DefId,
                   types: Vec<ty::t>)
                   -> Result<BuiltinBoundConditions,SelectionError>
        {
            // First check for markers and other nonsense.
            let tcx = this.tcx();
            match bound {
                ty::BoundSend => {
                    if
                        Some(def_id) == tcx.lang_items.no_send_bound() ||
                        Some(def_id) == tcx.lang_items.managed_bound()
                    {
                        return Err(Unimplemented);
                    }
                }

                ty::BoundCopy => {
                    if
                        Some(def_id) == tcx.lang_items.no_copy_bound() ||
                        Some(def_id) == tcx.lang_items.managed_bound() ||
                        ty::has_dtor(tcx, def_id)
                    {
                        return Err(Unimplemented);
                    }
                }

                ty::BoundSync => {
                    if
                        Some(def_id) == tcx.lang_items.no_sync_bound() ||
                        Some(def_id) == tcx.lang_items.managed_bound()
                    {
                        return Err(Unimplemented);
                    } else if
                        Some(def_id) == tcx.lang_items.unsafe_type()
                    {
                        // FIXME(#13231) -- we currently consider `UnsafeCell<T>`
                        // to always be sync. This is allow for types like `Queue`
                        // and `Mutex`, where `Queue<T> : Sync` is `T : Send`.
                        return Ok(If(Vec::new()));
                    }
                }

                ty::BoundSized => { }
            }

            Ok(If(types))
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
                         -> Result<Selection,SelectionError>
    {
        debug!("confirm_candidate({}, {})",
               obligation.repr(self.tcx()),
               candidate.repr(self.tcx()));

        match candidate {
            // FIXME -- see assemble_builtin_bound_candidates()
            BuiltinCandidate(_) |
            ErrorCandidate => {
                Ok(VtableBuiltin(VtableBuiltinData { nested: VecPerParamSpace::empty() }))
            }

            ParamCandidate(param) => {
                Ok(VtableParam(
                    try!(self.confirm_param_candidate(obligation, param))))
            }

            ImplCandidate(impl_def_id) => {
                let vtable_impl =
                    try!(self.confirm_impl_candidate(obligation, impl_def_id));
                Ok(VtableImpl(vtable_impl))
            }

            UnboxedClosureCandidate(closure_def_id) => {
                try!(self.confirm_unboxed_closure_candidate(obligation, closure_def_id));
                Ok(VtableUnboxedClosure(closure_def_id))
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

    fn confirm_builtin_candidate(&mut self,
                                 obligation: &Obligation,
                                 bound: ty::BuiltinBound)
                                 -> Result<VtableBuiltinData<Obligation>,SelectionError>
    {
        debug!("confirm_builtin_candidate({})",
               obligation.repr(self.tcx()));

        match try!(self.builtin_bound(bound, obligation.self_ty())) {
            If(nested) => Ok(self.vtable_builtin_data(obligation, bound, nested)),
            AmbiguousBuiltin |
            ParameterBuiltin => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    format!("builtin bound for {} was ambig",
                            obligation.repr(self.tcx())).as_slice());
            }
        }
    }

    fn vtable_builtin_data(&mut self,
                           obligation: &Obligation,
                           bound: ty::BuiltinBound,
                           nested: Vec<ty::t>)
                           -> VtableBuiltinData<Obligation>
    {
        let obligations =
            result::collect(
                nested
                    .iter()
                    .map(|&t| {
                        util::obligation_for_builtin_bound(
                            self.tcx(),
                            obligation.cause,
                            bound,
                            obligation.recursion_depth + 1,
                            t)
                    }));
        let obligations = match obligations {
            Ok(o) => o,
            Err(ErrorReported) => Vec::new()
        };
        let obligations = VecPerParamSpace::new(obligations, Vec::new(), Vec::new());
        VtableBuiltinData { nested: obligations }
    }

    fn confirm_impl_candidate(&mut self,
                              obligation: &Obligation,
                              impl_def_id: ast::DefId)
                              -> Result<VtableImplData<Obligation>,SelectionError>
    {
        debug!("confirm_impl_candidate({},{})",
               obligation.repr(self.tcx()),
               impl_def_id.repr(self.tcx()));

        // First, create the substitutions by matching the impl again,
        // this time not in a probe.
        let substs = self.rematch_impl(impl_def_id, obligation);
        Ok(self.vtable_impl(impl_def_id, substs, obligation.cause, obligation.recursion_depth + 1))
    }

    fn vtable_impl(&mut self,
                   impl_def_id: ast::DefId,
                   substs: Substs,
                   cause: ObligationCause,
                   recursion_depth: uint)
                   -> VtableImplData<Obligation>
    {
        let impl_obligations =
            self.impl_obligations(cause,
                                  recursion_depth,
                                  impl_def_id,
                                  &substs);
        VtableImplData { impl_def_id: impl_def_id,
                         substs: substs,
                         nested: impl_obligations }
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

    fn rematch_impl(&mut self,
                    impl_def_id: ast::DefId,
                    obligation: &Obligation)
                    -> Substs
    {
        match self.match_impl(impl_def_id, obligation) {
            Ok(substs) => {
                substs
            }
            Err(()) => {
                self.tcx().sess.bug(
                    format!("Impl {} was matchable against {} but now is not",
                            impl_def_id.repr(self.tcx()),
                            obligation.repr(self.tcx()))
                        .as_slice());
            }
        }
    }

    fn match_impl(&mut self,
                  impl_def_id: ast::DefId,
                  obligation: &Obligation)
                  -> Result<Substs, ()>
    {
        let impl_substs = util::fresh_substs_for_impl(self.infcx,
                                                      obligation.cause.span,
                                                      impl_def_id);

        let impl_trait_ref = ty::impl_trait_ref(self.tcx(),
                                                impl_def_id).unwrap();
        let impl_trait_ref = impl_trait_ref.subst(self.tcx(),
                                                  &impl_substs);

        match self.match_trait_refs(obligation, impl_trait_ref) {
            Ok(()) => Ok(impl_substs),
            Err(()) => Err(())
        }
    }

    fn match_trait_refs(&mut self,
                        obligation: &Obligation,
                        trait_ref: Rc<ty::TraitRef>)
                        -> Result<(),()>
    {
        let origin = infer::RelateOutputImplTypes(obligation.cause.span);
        match self.infcx.sub_trait_refs(false,
                                        origin,
                                        trait_ref,
                                        obligation.trait_ref.clone()) {
            Ok(()) => Ok(()),
            Err(_) => Err(()),
        }
    }

    fn match_inherent_impl(&mut self,
                           impl_def_id: ast::DefId,
                           obligation_cause: ObligationCause,
                           obligation_self_ty: ty::t)
                           -> Result<Substs,()>
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
            Ok(()) => {
                debug!("Matched impl_substs={}", impl_substs.repr(self.tcx()));
                Ok(impl_substs)
            }
            Err(()) => {
                debug!("NoMatch");
                Err(())
            }
        }
    }

    fn match_self_types(&mut self,
                        cause: ObligationCause,

                        // The self type provided by the impl/caller-obligation:
                        provided_self_ty: ty::t,

                        // The self type the obligation is for:
                        required_self_ty: ty::t)
                        -> Result<(),()>
    {
        // FIXME(#5781) -- equating the types is stronger than
        // necessary. Should consider variance of trait w/r/t Self.

        let origin = infer::RelateSelfType(cause.span);
        match self.infcx.eq_types(false,
                                  origin,
                                  provided_self_ty,
                                  required_self_ty) {
            Ok(()) => Ok(()),
            Err(_) => Err(()),
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

    fn push_stack<'o,'s:'o>(&mut self,
                            previous_stack: Option<&'s ObligationStack<'s>>,
                            obligation: &'o Obligation)
                            -> ObligationStack<'o>
    {
        let skol_trait_ref = obligation.trait_ref.fold_with(&mut self.skolemizer);

        ObligationStack {
            obligation: obligation,
            skol_trait_ref: skol_trait_ref,
            previous: previous_stack.map(|p| p), // FIXME variance
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

impl Repr for Candidate {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            ErrorCandidate => format!("ErrorCandidate"),
            BuiltinCandidate(b) => format!("BuiltinCandidate({})", b),
            UnboxedClosureCandidate(c) => format!("MatchedUnboxedClosureCandidate({})", c),
            ParamCandidate(ref a) => format!("ParamCandidate({})", a.repr(tcx)),
            ImplCandidate(a) => format!("ImplCandidate({})", a.repr(tcx)),
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
        format!("ObligationStack({})",
                self.obligation.repr(tcx))
    }
}

impl EvaluationResult {
    fn may_apply(&self) -> bool {
        match *self {
            EvaluatedToOk | EvaluatedToAmbig => true,
            EvaluatedToErr => false,
        }
    }
}
