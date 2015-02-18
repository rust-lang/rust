// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See `doc.rs` for high-level documentation
#![allow(dead_code)] // FIXME -- just temporarily

pub use self::MethodMatchResult::*;
pub use self::MethodMatchedData::*;
use self::SelectionCandidate::*;
use self::BuiltinBoundConditions::*;
use self::EvaluationResult::*;

use super::{DerivedObligationCause};
use super::{project};
use super::project::Normalized;
use super::{PredicateObligation, TraitObligation, ObligationCause};
use super::{ObligationCauseCode, BuiltinDerivedObligation};
use super::{SelectionError, Unimplemented, Overflow, OutputTypeParameterMismatch};
use super::{Selection};
use super::{SelectionResult};
use super::{VtableBuiltin, VtableImpl, VtableParam, VtableClosure,
            VtableFnPointer, VtableObject};
use super::{VtableImplData, VtableObjectData, VtableBuiltinData};
use super::object_safety;
use super::{util};

use middle::fast_reject;
use middle::subst::{Subst, Substs, TypeSpace, VecPerParamSpace};
use middle::ty::{self, RegionEscape, ToPolyTraitRef, Ty};
use middle::infer;
use middle::infer::{InferCtxt, TypeFreshener};
use middle::ty_fold::TypeFoldable;
use std::cell::RefCell;
use std::collections::hash_map::HashMap;
use std::rc::Rc;
use syntax::{abi, ast};
use util::common::ErrorReported;
use util::ppaux::Repr;

pub struct SelectionContext<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    closure_typer: &'cx (ty::ClosureTyper<'tcx>+'cx),

    /// Freshener used specifically for skolemizing entries on the
    /// obligation stack. This ensures that all entries on the stack
    /// at one time will have the same set of skolemized entries,
    /// which is important for checking for trait bounds that
    /// recursively require themselves.
    freshener: TypeFreshener<'cx, 'tcx>,

    /// If true, indicates that the evaluation should be conservative
    /// and consider the possibility of types outside this crate.
    /// This comes up primarily when resolving ambiguity. Imagine
    /// there is some trait reference `$0 : Bar` where `$0` is an
    /// inference variable. If `intercrate` is true, then we can never
    /// say for sure that this reference is not implemented, even if
    /// there are *no impls at all for `Bar`*, because `$0` could be
    /// bound to some type that in a downstream crate that implements
    /// `Bar`. This is the suitable mode for coherence. Elsewhere,
    /// though, we set this to false, because we are only interested
    /// in types that the user could actually have written --- in
    /// other words, we consider `$0 : Bar` to be unimplemented if
    /// there is no type that the user could *actually name* that
    /// would satisfy it. This avoids crippling inference, basically.
    intercrate: bool,
}

// A stack that walks back up the stack frame.
struct TraitObligationStack<'prev, 'tcx: 'prev> {
    obligation: &'prev TraitObligation<'tcx>,

    /// Trait ref from `obligation` but skolemized with the
    /// selection-context's freshener. Used to check for recursion.
    fresh_trait_ref: ty::PolyTraitRef<'tcx>,

    previous: Option<&'prev TraitObligationStack<'prev, 'tcx>>
}

#[derive(Clone)]
pub struct SelectionCache<'tcx> {
    hashmap: RefCell<HashMap<Rc<ty::TraitRef<'tcx>>,
                             SelectionResult<'tcx, SelectionCandidate<'tcx>>>>,
}

pub enum MethodMatchResult {
    MethodMatched(MethodMatchedData),
    MethodAmbiguous(/* list of impls that could apply */ Vec<ast::DefId>),
    MethodDidNotMatch,
}

#[derive(Copy, Debug)]
pub enum MethodMatchedData {
    // In the case of a precise match, we don't really need to store
    // how the match was found. So don't.
    PreciseMethodMatch,

    // In the case of a coercion, we need to know the precise impl so
    // that we can determine the type to which things were coerced.
    CoerciveMethodMatch(/* impl we matched */ ast::DefId)
}

/// The selection process begins by considering all impls, where
/// clauses, and so forth that might resolve an obligation.  Sometimes
/// we'll be able to say definitively that (e.g.) an impl does not
/// apply to the obligation: perhaps it is defined for `uint` but the
/// obligation is for `int`. In that case, we drop the impl out of the
/// list.  But the other cases are considered *candidates*.
///
/// Candidates can either be definitive or ambiguous. An ambiguous
/// candidate is one that might match or might not, depending on how
/// type variables wind up being resolved. This only occurs during inference.
///
/// For selection to succeed, there must be exactly one non-ambiguous
/// candidate.  Usually, it is not possible to have more than one
/// definitive candidate, due to the coherence rules. However, there is
/// one case where it could occur: if there is a blanket impl for a
/// trait (that is, an impl applied to all T), and a type parameter
/// with a where clause. In that case, we can have a candidate from the
/// where clause and a second candidate from the impl. This is not a
/// problem because coherence guarantees us that the impl which would
/// be used to satisfy the where clause is the same one that we see
/// now. To resolve this issue, therefore, we ignore impls if we find a
/// matching where clause. Part of the reason for this is that where
/// clauses can give additional information (like, the types of output
/// parameters) that would have to be inferred from the impl.
#[derive(PartialEq,Eq,Debug,Clone)]
enum SelectionCandidate<'tcx> {
    PhantomFnCandidate,
    BuiltinCandidate(ty::BuiltinBound),
    ParamCandidate(ty::PolyTraitRef<'tcx>),
    ImplCandidate(ast::DefId),

    /// This is a trait matching with a projected type as `Self`, and
    /// we found an applicable bound in the trait definition.
    ProjectionCandidate,

    /// Implementation of a `Fn`-family trait by one of the
    /// anonymous types generated for a `||` expression.
    ClosureCandidate(/* closure */ ast::DefId, Substs<'tcx>),

    /// Implementation of a `Fn`-family trait by one of the anonymous
    /// types generated for a fn pointer type (e.g., `fn(int)->int`)
    FnPointerCandidate,

    ObjectCandidate,

    ErrorCandidate,
}

struct SelectionCandidateSet<'tcx> {
    // a list of candidates that definitely apply to the current
    // obligation (meaning: types unify).
    vec: Vec<SelectionCandidate<'tcx>>,

    // if this is true, then there were candidates that might or might
    // not have applied, but we couldn't tell. This occurs when some
    // of the input types are type variables, in which case there are
    // various "builtin" rules that might or might not trigger.
    ambiguous: bool,
}

enum BuiltinBoundConditions<'tcx> {
    If(Vec<Ty<'tcx>>),
    ParameterBuiltin,
    AmbiguousBuiltin
}

#[derive(Debug)]
enum EvaluationResult<'tcx> {
    EvaluatedToOk,
    EvaluatedToAmbig,
    EvaluatedToErr(SelectionError<'tcx>),
}

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    pub fn new(infcx: &'cx InferCtxt<'cx, 'tcx>,
               closure_typer: &'cx ty::ClosureTyper<'tcx>)
               -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
            closure_typer: closure_typer,
            freshener: infcx.freshener(),
            intercrate: false,
        }
    }

    pub fn intercrate(infcx: &'cx InferCtxt<'cx, 'tcx>,
                      closure_typer: &'cx ty::ClosureTyper<'tcx>)
                      -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
            closure_typer: closure_typer,
            freshener: infcx.freshener(),
            intercrate: true,
        }
    }

    pub fn infcx(&self) -> &'cx InferCtxt<'cx, 'tcx> {
        self.infcx
    }

    pub fn tcx(&self) -> &'cx ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    pub fn param_env(&self) -> &'cx ty::ParameterEnvironment<'cx, 'tcx> {
        self.closure_typer.param_env()
    }

    pub fn closure_typer(&self) -> &'cx (ty::ClosureTyper<'tcx>+'cx) {
        self.closure_typer
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

    /// Attempts to satisfy the obligation. If successful, this will affect the surrounding
    /// type environment by performing unification.
    pub fn select(&mut self, obligation: &TraitObligation<'tcx>)
                  -> SelectionResult<'tcx, Selection<'tcx>> {
        debug!("select({})", obligation.repr(self.tcx()));
        assert!(!obligation.predicate.has_escaping_regions());

        let stack = self.push_stack(None, obligation);
        match try!(self.candidate_from_obligation(&stack)) {
            None => {
                self.consider_unification_despite_ambiguity(obligation);
                Ok(None)
            }
            Some(candidate) => Ok(Some(try!(self.confirm_candidate(obligation, candidate)))),
        }
    }

    /// In the particular case of unboxed closure obligations, we can
    /// sometimes do some amount of unification for the
    /// argument/return types even though we can't yet fully match obligation.
    /// The particular case we are interesting in is an obligation of the form:
    ///
    ///    C : FnFoo<A>
    ///
    /// where `C` is an unboxed closure type and `FnFoo` is one of the
    /// `Fn` traits. Because we know that users cannot write impls for closure types
    /// themselves, the only way that `C : FnFoo` can fail to match is under two
    /// conditions:
    ///
    /// 1. The closure kind for `C` is not yet known, because inference isn't complete.
    /// 2. The closure kind for `C` *is* known, but doesn't match what is needed.
    ///    For example, `C` may be a `FnOnce` closure, but a `Fn` closure is needed.
    ///
    /// In either case, we always know what argument types are
    /// expected by `C`, no matter what kind of `Fn` trait it
    /// eventually matches. So we can go ahead and unify the argument
    /// types, even though the end result is ambiguous.
    ///
    /// Note that this is safe *even if* the trait would never be
    /// matched (case 2 above). After all, in that case, an error will
    /// result, so it kind of doesn't matter what we do --- unifying
    /// the argument types can only be helpful to the user, because
    /// once they patch up the kind of closure that is expected, the
    /// argment types won't really change.
    fn consider_unification_despite_ambiguity(&mut self, obligation: &TraitObligation<'tcx>) {
        // Is this a `C : FnFoo(...)` trait reference for some trait binding `FnFoo`?
        match self.tcx().lang_items.fn_trait_kind(obligation.predicate.0.def_id()) {
            Some(_) => { }
            None => { return; }
        }

        // Is the self-type a closure type? We ignore bindings here
        // because if it is a closure type, it must be a closure type from
        // within this current fn, and hence none of the higher-ranked
        // lifetimes can appear inside the self-type.
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        let (closure_def_id, substs) = match self_ty.sty {
            ty::ty_closure(id, _, ref substs) => (id, substs.clone()),
            _ => { return; }
        };
        assert!(!substs.has_escaping_regions());

        let closure_trait_ref = self.closure_trait_ref(obligation, closure_def_id, substs);
        match self.confirm_poly_trait_refs(obligation.cause.clone(),
                                           obligation.predicate.to_poly_trait_ref(),
                                           closure_trait_ref) {
            Ok(()) => { }
            Err(_) => { /* Silently ignore errors. */ }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // EVALUATION
    //
    // Tests whether an obligation can be selected or whether an impl
    // can be applied to particular types. It skips the "confirmation"
    // step and hence completely ignores output type parameters.
    //
    // The result is "true" if the obligation *may* hold and "false" if
    // we can be sure it does not.

    /// Evaluates whether the obligation `obligation` can be satisfied (by any means).
    pub fn evaluate_obligation(&mut self,
                               obligation: &PredicateObligation<'tcx>)
                               -> bool
    {
        debug!("evaluate_obligation({})",
               obligation.repr(self.tcx()));

        self.evaluate_predicate_recursively(None, obligation).may_apply()
    }

    fn evaluate_builtin_bound_recursively<'o>(&mut self,
                                              bound: ty::BuiltinBound,
                                              previous_stack: &TraitObligationStack<'o, 'tcx>,
                                              ty: Ty<'tcx>)
                                              -> EvaluationResult<'tcx>
    {
        let obligation =
            util::predicate_for_builtin_bound(
                self.tcx(),
                previous_stack.obligation.cause.clone(),
                bound,
                previous_stack.obligation.recursion_depth + 1,
                ty);

        match obligation {
            Ok(obligation) => {
                self.evaluate_predicate_recursively(Some(previous_stack), &obligation)
            }
            Err(ErrorReported) => {
                EvaluatedToOk
            }
        }
    }

    fn evaluate_predicates_recursively<'a,'o,I>(&mut self,
                                                stack: Option<&TraitObligationStack<'o, 'tcx>>,
                                                predicates: I)
                                                -> EvaluationResult<'tcx>
        where I : Iterator<Item=&'a PredicateObligation<'tcx>>, 'tcx:'a
    {
        let mut result = EvaluatedToOk;
        for obligation in predicates {
            match self.evaluate_predicate_recursively(stack, obligation) {
                EvaluatedToErr(e) => { return EvaluatedToErr(e); }
                EvaluatedToAmbig => { result = EvaluatedToAmbig; }
                EvaluatedToOk => { }
            }
        }
        result
    }

    fn evaluate_predicate_recursively<'o>(&mut self,
                                          previous_stack: Option<&TraitObligationStack<'o, 'tcx>>,
                                          obligation: &PredicateObligation<'tcx>)
                                           -> EvaluationResult<'tcx>
    {
        debug!("evaluate_predicate_recursively({})",
               obligation.repr(self.tcx()));

        match obligation.predicate {
            ty::Predicate::Trait(ref t) => {
                assert!(!t.has_escaping_regions());
                let obligation = obligation.with(t.clone());
                self.evaluate_obligation_recursively(previous_stack, &obligation)
            }

            ty::Predicate::Equate(ref p) => {
                let result = self.infcx.probe(|_| {
                    self.infcx.equality_predicate(obligation.cause.span, p)
                });
                match result {
                    Ok(()) => EvaluatedToOk,
                    Err(_) => EvaluatedToErr(Unimplemented),
                }
            }

            ty::Predicate::TypeOutlives(..) | ty::Predicate::RegionOutlives(..) => {
                // we do not consider region relationships when
                // evaluating trait matches
                EvaluatedToOk
            }

            ty::Predicate::Projection(ref data) => {
                self.infcx.probe(|_| {
                    let project_obligation = obligation.with(data.clone());
                    match project::poly_project_and_unify_type(self, &project_obligation) {
                        Ok(Some(subobligations)) => {
                            self.evaluate_predicates_recursively(previous_stack,
                                                                 subobligations.iter())
                        }
                        Ok(None) => {
                            EvaluatedToAmbig
                        }
                        Err(_) => {
                            EvaluatedToErr(Unimplemented)
                        }
                    }
                })
            }
        }
    }

    fn evaluate_obligation_recursively<'o>(&mut self,
                                           previous_stack: Option<&TraitObligationStack<'o, 'tcx>>,
                                           obligation: &TraitObligation<'tcx>)
                                           -> EvaluationResult<'tcx>
    {
        debug!("evaluate_obligation_recursively({})",
               obligation.repr(self.tcx()));

        let stack = self.push_stack(previous_stack.map(|x| x), obligation);

        let result = self.evaluate_stack(&stack);

        debug!("result: {:?}", result);
        result
    }

    fn evaluate_stack<'o>(&mut self,
                          stack: &TraitObligationStack<'o, 'tcx>)
                          -> EvaluationResult<'tcx>
    {
        // In intercrate mode, whenever any of the types are unbound,
        // there can always be an impl. Even if there are no impls in
        // this crate, perhaps the type would be unified with
        // something from another crate that does provide an impl.
        //
        // In intracrate mode, we must still be conservative. The reason is
        // that we want to avoid cycles. Imagine an impl like:
        //
        //     impl<T:Eq> Eq for Vec<T>
        //
        // and a trait reference like `$0 : Eq` where `$0` is an
        // unbound variable. When we evaluate this trait-reference, we
        // will unify `$0` with `Vec<$1>` (for some fresh variable
        // `$1`), on the condition that `$1 : Eq`. We will then wind
        // up with many candidates (since that are other `Eq` impls
        // that apply) and try to winnow things down. This results in
        // a recursive evaluation that `$1 : Eq` -- as you can
        // imagine, this is just where we started. To avoid that, we
        // check for unbound variables and return an ambiguous (hence possible)
        // match if we've seen this trait before.
        //
        // This suffices to allow chains like `FnMut` implemented in
        // terms of `Fn` etc, but we could probably make this more
        // precise still.
        let input_types = stack.fresh_trait_ref.0.input_types();
        let unbound_input_types = input_types.iter().any(|&t| ty::type_is_fresh(t));
        if
            unbound_input_types &&
             (self.intercrate ||
              stack.iter().skip(1).any(
                  |prev| stack.fresh_trait_ref.def_id() == prev.fresh_trait_ref.def_id()))
        {
            debug!("evaluate_stack({}) --> unbound argument, recursion -->  ambiguous",
                   stack.fresh_trait_ref.repr(self.tcx()));
            return EvaluatedToAmbig;
        }

        // If there is any previous entry on the stack that precisely
        // matches this obligation, then we can assume that the
        // obligation is satisfied for now (still all other conditions
        // must be met of course). One obvious case this comes up is
        // marker traits like `Send`. Think of a linked list:
        //
        //    struct List<T> { data: T, next: Option<Box<List<T>>> {
        //
        // `Box<List<T>>` will be `Send` if `T` is `Send` and
        // `Option<Box<List<T>>>` is `Send`, and in turn
        // `Option<Box<List<T>>>` is `Send` if `Box<List<T>>` is
        // `Send`.
        //
        // Note that we do this comparison using the `fresh_trait_ref`
        // fields. Because these have all been skolemized using
        // `self.freshener`, we can be sure that (a) this will not
        // affect the inferencer state and (b) that if we see two
        // skolemized types with the same index, they refer to the
        // same unbound type variable.
        if
            stack.iter()
            .skip(1) // skip top-most frame
            .any(|prev| stack.fresh_trait_ref == prev.fresh_trait_ref)
        {
            debug!("evaluate_stack({}) --> recursive",
                   stack.fresh_trait_ref.repr(self.tcx()));
            return EvaluatedToOk;
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.winnow_candidate(stack, &c),
            Ok(None) => EvaluatedToAmbig,
            Err(e) => EvaluatedToErr(e),
        }
    }

    /// Evaluates whether the impl with id `impl_def_id` could be applied to the self type
    /// `obligation_self_ty`. This can be used either for trait or inherent impls.
    pub fn evaluate_impl(&mut self,
                         impl_def_id: ast::DefId,
                         obligation: &TraitObligation<'tcx>)
                         -> bool
    {
        debug!("evaluate_impl(impl_def_id={}, obligation={})",
               impl_def_id.repr(self.tcx()),
               obligation.repr(self.tcx()));

        self.infcx.probe(|snapshot| {
            let (skol_obligation_trait_ref, skol_map) =
                self.infcx().skolemize_late_bound_regions(&obligation.predicate, snapshot);
            match self.match_impl(impl_def_id, obligation, snapshot,
                                  &skol_map, skol_obligation_trait_ref.trait_ref.clone()) {
                Ok(substs) => {
                    let vtable_impl = self.vtable_impl(impl_def_id,
                                                       substs,
                                                       obligation.cause.clone(),
                                                       obligation.recursion_depth + 1,
                                                       skol_map,
                                                       snapshot);
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

    fn candidate_from_obligation<'o>(&mut self,
                                     stack: &TraitObligationStack<'o, 'tcx>)
                                     -> SelectionResult<'tcx, SelectionCandidate<'tcx>>
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
        // separately rather than using `stack.fresh_trait_ref` -- this
        // is because we want the unbound variables to be replaced
        // with fresh skolemized types starting from index 0.
        let cache_fresh_trait_pred =
            self.infcx.freshen(stack.obligation.predicate.clone());
        debug!("candidate_from_obligation(cache_fresh_trait_pred={}, obligation={})",
               cache_fresh_trait_pred.repr(self.tcx()),
               stack.repr(self.tcx()));
        assert!(!stack.obligation.predicate.has_escaping_regions());

        match self.check_candidate_cache(&cache_fresh_trait_pred) {
            Some(c) => {
                debug!("CACHE HIT: cache_fresh_trait_pred={}, candidate={}",
                       cache_fresh_trait_pred.repr(self.tcx()),
                       c.repr(self.tcx()));
                return c;
            }
            None => { }
        }

        // If no match, compute result and insert into cache.
        let candidate = self.candidate_from_obligation_no_cache(stack);

        if self.should_update_candidate_cache(&cache_fresh_trait_pred, &candidate) {
            debug!("CACHE MISS: cache_fresh_trait_pred={}, candidate={}",
                   cache_fresh_trait_pred.repr(self.tcx()), candidate.repr(self.tcx()));
            self.insert_candidate_cache(cache_fresh_trait_pred, candidate.clone());
        }

        candidate
    }

    fn candidate_from_obligation_no_cache<'o>(&mut self,
                                              stack: &TraitObligationStack<'o, 'tcx>)
                                              -> SelectionResult<'tcx, SelectionCandidate<'tcx>>
    {
        if ty::type_is_error(stack.obligation.predicate.0.self_ty()) {
            return Ok(Some(ErrorCandidate));
        }

        let candidate_set = try!(self.assemble_candidates(stack));

        if candidate_set.ambiguous {
            debug!("candidate set contains ambig");
            return Ok(None);
        }

        let mut candidates = candidate_set.vec;

        debug!("assembled {} candidates for {}: {}",
               candidates.len(),
               stack.repr(self.tcx()),
               candidates.repr(self.tcx()));

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
                    (0..candidates.len())
                    .filter(|&j| i != j)
                    .any(|j| self.candidate_should_be_dropped_in_favor_of(&candidates[i],
                                                                          &candidates[j]));
                if is_dup {
                    debug!("Dropping candidate #{}/{}: {}",
                           i, candidates.len(), candidates[i].repr(self.tcx()));
                    candidates.swap_remove(i);
                } else {
                    debug!("Retaining candidate #{}/{}: {}",
                           i, candidates.len(), candidates[i].repr(self.tcx()));
                    i += 1;
                }
            }
        }

        // If there are *STILL* multiple candidates, give up and
        // report ambiguity.
        if candidates.len() > 1 {
            debug!("multiple matches, ambig");
            return Ok(None);
        }


        // If there are *NO* candidates, that there are no impls --
        // that we know of, anyway. Note that in the case where there
        // are unbound type variables within the obligation, it might
        // be the case that you could still satisfy the obligation
        // from another crate by instantiating the type variables with
        // a type from another crate that does have an impl. This case
        // is checked for in `evaluate_stack` (and hence users
        // who might care about this case, like coherence, should use
        // that function).
        if candidates.len() == 0 {
            return Err(Unimplemented);
        }

        // Just one candidate left.
        let candidate = candidates.pop().unwrap();

        match candidate {
            ImplCandidate(def_id) => {
                match ty::trait_impl_polarity(self.tcx(), def_id) {
                    Some(ast::ImplPolarity::Negative) => return Err(Unimplemented),
                    _ => {}
                }
            }
            _ => {}
        }

        Ok(Some(candidate))
    }

    fn pick_candidate_cache(&self) -> &SelectionCache<'tcx> {
        // If there are any where-clauses in scope, then we always use
        // a cache local to this particular scope. Otherwise, we
        // switch to a global cache. We used to try and draw
        // finer-grained distinctions, but that led to a serious of
        // annoying and weird bugs like #22019 and #18290. This simple
        // rule seems to be pretty clearly safe and also still retains
        // a very high hit rate (~95% when compiling rustc).
        if !self.param_env().caller_bounds.is_empty() {
            return &self.param_env().selection_cache;
        }

        // Avoid using the master cache during coherence and just rely
        // on the local cache. This effectively disables caching
        // during coherence. It is really just a simplification to
        // avoid us having to fear that coherence results "pollute"
        // the master cache. Since coherence executes pretty quickly,
        // it's not worth going to more trouble to increase the
        // hit-rate I don't think.
        if self.intercrate {
            return &self.param_env().selection_cache;
        }

        // Otherwise, we can use the global cache.
        &self.tcx().selection_cache
    }

    fn check_candidate_cache(&mut self,
                             cache_fresh_trait_pred: &ty::PolyTraitPredicate<'tcx>)
                             -> Option<SelectionResult<'tcx, SelectionCandidate<'tcx>>>
    {
        let cache = self.pick_candidate_cache();
        let hashmap = cache.hashmap.borrow();
        hashmap.get(&cache_fresh_trait_pred.0.trait_ref).cloned()
    }

    fn insert_candidate_cache(&mut self,
                              cache_fresh_trait_pred: ty::PolyTraitPredicate<'tcx>,
                              candidate: SelectionResult<'tcx, SelectionCandidate<'tcx>>)
    {
        let cache = self.pick_candidate_cache();
        let mut hashmap = cache.hashmap.borrow_mut();
        hashmap.insert(cache_fresh_trait_pred.0.trait_ref.clone(), candidate);
    }

    fn should_update_candidate_cache(&mut self,
                                     cache_fresh_trait_pred: &ty::PolyTraitPredicate<'tcx>,
                                     candidate: &SelectionResult<'tcx, SelectionCandidate<'tcx>>)
                                     -> bool
    {
        // In general, it's a good idea to cache results, even
        // ambiguous ones, to save us some trouble later. But we have
        // to be careful not to cache results that could be
        // invalidated later by advances in inference. Normally, this
        // is not an issue, because any inference variables whose
        // types are not yet bound are "freshened" in the cache key,
        // which means that if we later get the same request once that
        // type variable IS bound, we'll have a different cache key.
        // For example, if we have `Vec<_#0t> : Foo`, and `_#0t` is
        // not yet known, we may cache the result as `None`. But if
        // later `_#0t` is bound to `Bar`, then when we freshen we'll
        // have `Vec<Bar> : Foo` as the cache key.
        //
        // HOWEVER, it CAN happen that we get an ambiguity result in
        // one particular case around closures where the cache key
        // would not change. That is when the precise types of the
        // upvars that a closure references have not yet been figured
        // out (i.e., because it is not yet known if they are captured
        // by ref, and if by ref, what kind of ref). In these cases,
        // when matching a builtin bound, we will yield back an
        // ambiguous result. But the *cache key* is just the closure type,
        // it doesn't capture the state of the upvar computation.
        //
        // To avoid this trap, just don't cache ambiguous results if
        // the self-type contains no inference byproducts (that really
        // shouldn't happen in other circumstances anyway, given
        // coherence).

        match *candidate {
            Ok(Some(_)) | Err(_) => true,
            Ok(None) => {
                cache_fresh_trait_pred.0.input_types().iter().any(|&t| ty::type_has_ty_infer(t))
            }
        }
    }

    fn assemble_candidates<'o>(&mut self,
                               stack: &TraitObligationStack<'o, 'tcx>)
                               -> Result<SelectionCandidateSet<'tcx>, SelectionError<'tcx>>
    {
        let TraitObligationStack { obligation, .. } = *stack;

        let mut candidates = SelectionCandidateSet {
            vec: Vec::new(),
            ambiguous: false
        };

        // Check for the `PhantomFn` trait. This is really just a
        // special annotation that is *always* considered to match, no
        // matter what the type parameters are etc.
        if self.tcx().lang_items.phantom_fn() == Some(obligation.predicate.def_id()) {
            candidates.vec.push(PhantomFnCandidate);
            return Ok(candidates);
        }

        // Other bounds. Consider both in-scope bounds from fn decl
        // and applicable impls. There is a certain set of precedence rules here.

        match self.tcx().lang_items.to_builtin_kind(obligation.predicate.def_id()) {
            Some(ty::BoundCopy) => {
                debug!("obligation self ty is {}",
                       obligation.predicate.0.self_ty().repr(self.tcx()));

                try!(self.assemble_candidates_from_impls(obligation, &mut candidates));

                try!(self.assemble_builtin_bound_candidates(ty::BoundCopy,
                                                            stack,
                                                            &mut candidates));
            }
            Some(bound @ ty::BoundSend) |
            Some(bound @ ty::BoundSync) => {
                try!(self.assemble_candidates_from_impls(obligation, &mut candidates));

                // No explicit impls were declared for this type, consider the fallback rules.
                if candidates.vec.is_empty() && !candidates.ambiguous {
                    try!(self.assemble_builtin_bound_candidates(bound, stack, &mut candidates));
                }
            }

            Some(bound @ ty::BoundSized) => {
                // Sized and Copy are always automatically computed.
                try!(self.assemble_builtin_bound_candidates(bound, stack, &mut candidates));
            }

            None => {
                // For the time being, we ignore user-defined impls for builtin-bounds, other than
                // `Copy`.
                // (And unboxed candidates only apply to the Fn/FnMut/etc traits.)
                try!(self.assemble_closure_candidates(obligation, &mut candidates));
                try!(self.assemble_fn_pointer_candidates(obligation, &mut candidates));
                try!(self.assemble_candidates_from_impls(obligation, &mut candidates));
                self.assemble_candidates_from_object_ty(obligation, &mut candidates);
            }
        }

        self.assemble_candidates_from_projected_tys(obligation, &mut candidates);
        try!(self.assemble_candidates_from_caller_bounds(stack, &mut candidates));
        debug!("candidate list size: {}", candidates.vec.len());
        Ok(candidates)
    }

    fn assemble_candidates_from_projected_tys(&mut self,
                                              obligation: &TraitObligation<'tcx>,
                                              candidates: &mut SelectionCandidateSet<'tcx>)
    {
        let poly_trait_predicate =
            self.infcx().resolve_type_vars_if_possible(&obligation.predicate);

        debug!("assemble_candidates_for_projected_tys({},{})",
               obligation.repr(self.tcx()),
               poly_trait_predicate.repr(self.tcx()));

        // FIXME(#20297) -- just examining the self-type is very simplistic

        // before we go into the whole skolemization thing, just
        // quickly check if the self-type is a projection at all.
        let trait_def_id = match poly_trait_predicate.0.trait_ref.self_ty().sty {
            ty::ty_projection(ref data) => data.trait_ref.def_id,
            ty::ty_infer(ty::TyVar(_)) => {
                // If the self-type is an inference variable, then it MAY wind up
                // being a projected type, so induce an ambiguity.
                //
                // FIXME(#20297) -- being strict about this can cause
                // inference failures with BorrowFrom, which is
                // unfortunate. Can we do better here?
                debug!("assemble_candidates_for_projected_tys: ambiguous self-type");
                candidates.ambiguous = true;
                return;
            }
            _ => { return; }
        };

        debug!("assemble_candidates_for_projected_tys: trait_def_id={}",
               trait_def_id.repr(self.tcx()));

        let result = self.infcx.probe(|snapshot| {
            self.match_projection_obligation_against_bounds_from_trait(obligation,
                                                                       snapshot)
        });

        if result {
            candidates.vec.push(ProjectionCandidate);
        }
    }

    fn match_projection_obligation_against_bounds_from_trait(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        snapshot: &infer::CombinedSnapshot)
        -> bool
    {
        let poly_trait_predicate =
            self.infcx().resolve_type_vars_if_possible(&obligation.predicate);
        let (skol_trait_predicate, skol_map) =
            self.infcx().skolemize_late_bound_regions(&poly_trait_predicate, snapshot);
        debug!("match_projection_obligation_against_bounds_from_trait: \
                skol_trait_predicate={} skol_map={}",
               skol_trait_predicate.repr(self.tcx()),
               skol_map.repr(self.tcx()));

        let projection_trait_ref = match skol_trait_predicate.trait_ref.self_ty().sty {
            ty::ty_projection(ref data) => &data.trait_ref,
            _ => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    &format!("match_projection_obligation_against_bounds_from_trait() called \
                              but self-ty not a projection: {}",
                             skol_trait_predicate.trait_ref.self_ty().repr(self.tcx())));
            }
        };
        debug!("match_projection_obligation_against_bounds_from_trait: \
                projection_trait_ref={}",
               projection_trait_ref.repr(self.tcx()));

        let trait_predicates = ty::lookup_predicates(self.tcx(), projection_trait_ref.def_id);
        let bounds = trait_predicates.instantiate(self.tcx(), projection_trait_ref.substs);
        debug!("match_projection_obligation_against_bounds_from_trait: \
                bounds={}",
               bounds.repr(self.tcx()));

        let matching_bound =
            util::elaborate_predicates(self.tcx(), bounds.predicates.into_vec())
            .filter_to_traits()
            .find(
                |bound| self.infcx.probe(
                    |_| self.match_projection(obligation,
                                              bound.clone(),
                                              skol_trait_predicate.trait_ref.clone(),
                                              &skol_map,
                                              snapshot)));

        debug!("match_projection_obligation_against_bounds_from_trait: \
                matching_bound={}",
               matching_bound.repr(self.tcx()));
        match matching_bound {
            None => false,
            Some(bound) => {
                // Repeat the successful match, if any, this time outside of a probe.
                let result = self.match_projection(obligation,
                                                   bound,
                                                   skol_trait_predicate.trait_ref.clone(),
                                                   &skol_map,
                                                   snapshot);
                assert!(result);
                true
            }
        }
    }

    fn match_projection(&mut self,
                        obligation: &TraitObligation<'tcx>,
                        trait_bound: ty::PolyTraitRef<'tcx>,
                        skol_trait_ref: Rc<ty::TraitRef<'tcx>>,
                        skol_map: &infer::SkolemizationMap,
                        snapshot: &infer::CombinedSnapshot)
                        -> bool
    {
        assert!(!skol_trait_ref.has_escaping_regions());
        let origin = infer::RelateOutputImplTypes(obligation.cause.span);
        match self.infcx.sub_poly_trait_refs(false,
                                             origin,
                                             trait_bound.clone(),
                                             ty::Binder(skol_trait_ref.clone())) {
            Ok(()) => { }
            Err(_) => { return false; }
        }

        self.infcx.leak_check(skol_map, snapshot).is_ok()
    }

    /// Given an obligation like `<SomeTrait for T>`, search the obligations that the caller
    /// supplied to find out whether it is listed among them.
    ///
    /// Never affects inference environment.
    fn assemble_candidates_from_caller_bounds<'o>(&mut self,
                                                  stack: &TraitObligationStack<'o, 'tcx>,
                                                  candidates: &mut SelectionCandidateSet<'tcx>)
                                                  -> Result<(),SelectionError<'tcx>>
    {
        debug!("assemble_candidates_from_caller_bounds({})",
               stack.obligation.repr(self.tcx()));

        let caller_trait_refs: Vec<_> =
            self.param_env().caller_bounds.iter()
            .filter_map(|o| o.to_opt_poly_trait_ref())
            .collect();

        let all_bounds =
            util::transitive_bounds(
                self.tcx(), &caller_trait_refs[..]);

        let matching_bounds =
            all_bounds.filter(
                |bound| self.evaluate_where_clause(stack, bound.clone()).may_apply());

        let param_candidates =
            matching_bounds.map(|bound| ParamCandidate(bound));

        candidates.vec.extend(param_candidates);

        Ok(())
    }

    fn evaluate_where_clause<'o>(&mut self,
                                 stack: &TraitObligationStack<'o, 'tcx>,
                                 where_clause_trait_ref: ty::PolyTraitRef<'tcx>)
                                 -> EvaluationResult<'tcx>
    {
        self.infcx().probe(move |_| {
            match self.match_where_clause_trait_ref(stack.obligation, where_clause_trait_ref) {
                Ok(obligations) => {
                    self.evaluate_predicates_recursively(Some(stack), obligations.iter())
                }
                Err(()) => {
                    EvaluatedToErr(Unimplemented)
                }
            }
        })
    }

    /// Check for the artificial impl that the compiler will create for an obligation like `X :
    /// FnMut<..>` where `X` is a closure type.
    ///
    /// Note: the type parameters on a closure candidate are modeled as *output* type
    /// parameters and hence do not affect whether this trait is a match or not. They will be
    /// unified during the confirmation step.
    fn assemble_closure_candidates(&mut self,
                                   obligation: &TraitObligation<'tcx>,
                                   candidates: &mut SelectionCandidateSet<'tcx>)
                                   -> Result<(),SelectionError<'tcx>>
    {
        let kind = match self.tcx().lang_items.fn_trait_kind(obligation.predicate.0.def_id()) {
            Some(k) => k,
            None => { return Ok(()); }
        };

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        let (closure_def_id, substs) = match self_ty.sty {
            ty::ty_closure(id, _, ref substs) => (id, substs.clone()),
            ty::ty_infer(ty::TyVar(_)) => {
                debug!("assemble_unboxed_closure_candidates: ambiguous self-type");
                candidates.ambiguous = true;
                return Ok(());
            }
            _ => { return Ok(()); }
        };

        debug!("assemble_unboxed_candidates: self_ty={} kind={:?} obligation={}",
               self_ty.repr(self.tcx()),
               kind,
               obligation.repr(self.tcx()));

        match self.closure_typer.closure_kind(closure_def_id) {
            Some(closure_kind) => {
                debug!("assemble_unboxed_candidates: closure_kind = {:?}", closure_kind);
                if closure_kind == kind {
                    candidates.vec.push(ClosureCandidate(closure_def_id, substs.clone()));
                }
            }
            None => {
                debug!("assemble_unboxed_candidates: closure_kind not yet known");
                candidates.ambiguous = true;
            }
        }

        Ok(())
    }

    /// Implement one of the `Fn()` family for a fn pointer.
    fn assemble_fn_pointer_candidates(&mut self,
                                      obligation: &TraitObligation<'tcx>,
                                      candidates: &mut SelectionCandidateSet<'tcx>)
                                      -> Result<(),SelectionError<'tcx>>
    {
        // We provide a `Fn` impl for fn pointers. There is no need to provide
        // the other traits (e.g. `FnMut`) since those are provided by blanket
        // impls.
        if Some(obligation.predicate.def_id()) != self.tcx().lang_items.fn_trait() {
            return Ok(());
        }

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        match self_ty.sty {
            ty::ty_infer(ty::TyVar(_)) => {
                debug!("assemble_fn_pointer_candidates: ambiguous self-type");
                candidates.ambiguous = true; // could wind up being a fn() type
            }

            // provide an impl, but only for suitable `fn` pointers
            ty::ty_bare_fn(_, &ty::BareFnTy {
                unsafety: ast::Unsafety::Normal,
                abi: abi::Rust,
                sig: ty::Binder(ty::FnSig {
                    inputs: _,
                    output: ty::FnConverging(_),
                    variadic: false
                })
            }) => {
                candidates.vec.push(FnPointerCandidate);
            }

            _ => { }
        }

        Ok(())
    }

    /// Search for impls that might apply to `obligation`.
    fn assemble_candidates_from_impls(&mut self,
                                      obligation: &TraitObligation<'tcx>,
                                      candidates: &mut SelectionCandidateSet<'tcx>)
                                      -> Result<(), SelectionError<'tcx>>
    {
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        debug!("assemble_candidates_from_impls(self_ty={})", self_ty.repr(self.tcx()));

        let all_impls = self.all_impls(obligation.predicate.def_id());
        for &impl_def_id in &all_impls {
            self.infcx.probe(|snapshot| {
                let (skol_obligation_trait_pred, skol_map) =
                    self.infcx().skolemize_late_bound_regions(&obligation.predicate, snapshot);
                match self.match_impl(impl_def_id, obligation, snapshot,
                                      &skol_map, skol_obligation_trait_pred.trait_ref.clone()) {
                    Ok(_) => {
                        candidates.vec.push(ImplCandidate(impl_def_id));
                    }
                    Err(()) => { }
                }
            });
        }
        Ok(())
    }

    /// Search for impls that might apply to `obligation`.
    fn assemble_candidates_from_object_ty(&mut self,
                                          obligation: &TraitObligation<'tcx>,
                                          candidates: &mut SelectionCandidateSet<'tcx>)
    {
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());

        debug!("assemble_candidates_from_object_ty(self_ty={})",
               self_ty.repr(self.tcx()));

        // Object-safety candidates are only applicable to object-safe
        // traits. Including this check is useful because it helps
        // inference in cases of traits like `BorrowFrom`, which are
        // not object-safe, and which rely on being able to infer the
        // self-type from one of the other inputs. Without this check,
        // these cases wind up being considered ambiguous due to a
        // (spurious) ambiguity introduced here.
        if !object_safety::is_object_safe(self.tcx(), obligation.predicate.to_poly_trait_ref()) {
            return;
        }

        let poly_trait_ref = match self_ty.sty {
            ty::ty_trait(ref data) => {
                data.principal_trait_ref_with_self_ty(self.tcx(), self_ty)
            }
            ty::ty_infer(ty::TyVar(_)) => {
                debug!("assemble_candidates_from_object_ty: ambiguous");
                candidates.ambiguous = true; // could wind up being an object type
                return;
            }
            _ => {
                return;
            }
        };

        debug!("assemble_candidates_from_object_ty: poly_trait_ref={}",
               poly_trait_ref.repr(self.tcx()));

        // see whether the object trait can be upcast to the trait we are looking for
        let obligation_def_id = obligation.predicate.def_id();
        let upcast_trait_ref = match util::upcast(self.tcx(), poly_trait_ref, obligation_def_id) {
            Some(r) => r,
            None => { return; }
        };

        debug!("assemble_candidates_from_object_ty: upcast_trait_ref={}",
               upcast_trait_ref.repr(self.tcx()));

        // check whether the upcast version of the trait-ref matches what we are looking for
        if let Ok(()) = self.infcx.probe(|_| self.match_poly_trait_ref(obligation,
                                                                       upcast_trait_ref.clone())) {
            debug!("assemble_candidates_from_object_ty: matched, pushing candidate");
            candidates.vec.push(ObjectCandidate);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // WINNOW
    //
    // Winnowing is the process of attempting to resolve ambiguity by
    // probing further. During the winnowing process, we unify all
    // type variables (ignoring skolemization) and then we also
    // attempt to evaluate recursive bounds to see if they are
    // satisfied.

    /// Further evaluate `candidate` to decide whether all type parameters match and whether nested
    /// obligations are met. Returns true if `candidate` remains viable after this further
    /// scrutiny.
    fn winnow_candidate<'o>(&mut self,
                            stack: &TraitObligationStack<'o, 'tcx>,
                            candidate: &SelectionCandidate<'tcx>)
                            -> EvaluationResult<'tcx>
    {
        debug!("winnow_candidate: candidate={}", candidate.repr(self.tcx()));
        let result = self.infcx.probe(|_| {
            let candidate = (*candidate).clone();
            match self.confirm_candidate(stack.obligation, candidate) {
                Ok(selection) => self.winnow_selection(Some(stack), selection),
                Err(error) => EvaluatedToErr(error),
            }
        });
        debug!("winnow_candidate depth={} result={:?}",
               stack.obligation.recursion_depth, result);
        result
    }

    fn winnow_selection<'o>(&mut self,
                            stack: Option<&TraitObligationStack<'o, 'tcx>>,
                            selection: Selection<'tcx>)
                            -> EvaluationResult<'tcx>
    {
        self.evaluate_predicates_recursively(stack, selection.iter_nested())
    }

    /// Returns true if `candidate_i` should be dropped in favor of
    /// `candidate_j`.  Generally speaking we will drop duplicate
    /// candidates and prefer where-clause candidates.
    fn candidate_should_be_dropped_in_favor_of<'o>(&mut self,
                                                   candidate_i: &SelectionCandidate<'tcx>,
                                                   candidate_j: &SelectionCandidate<'tcx>)
                                                   -> bool
    {
        if candidate_i == candidate_j {
            return true;
        }

        match (candidate_i, candidate_j) {
            (&ImplCandidate(..), &ParamCandidate(..)) |
            (&ClosureCandidate(..), &ParamCandidate(..)) |
            (&FnPointerCandidate(..), &ParamCandidate(..)) |
            (&BuiltinCandidate(..), &ParamCandidate(..)) => {
                // We basically prefer always prefer to use a
                // where-clause over another option. Where clauses
                // impose the burden of finding the exact match onto
                // the caller. Using an impl in preference of a where
                // clause can also lead us to "overspecialize", as in
                // #18453.
                true
            }
            (&ProjectionCandidate, &ParamCandidate(_)) => {
                // FIXME(#20297) -- this gives where clauses precedent
                // over projections. Really these are just two means
                // of deducing information (one based on the where
                // clauses on the trait definition; one based on those
                // on the enclosing scope), and it'd be better to
                // integrate them more intelligently. But for now this
                // seems ok. If we DON'T give where clauses
                // precedence, we run into trouble in default methods,
                // where both the projection bounds for `Self::A` and
                // the where clauses are in scope.
                true
            }
            _ => {
                false
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

    fn assemble_builtin_bound_candidates<'o>(&mut self,
                                             bound: ty::BuiltinBound,
                                             stack: &TraitObligationStack<'o, 'tcx>,
                                             candidates: &mut SelectionCandidateSet<'tcx>)
                                             -> Result<(),SelectionError<'tcx>>
    {
        match self.builtin_bound(bound, stack.obligation) {
            Ok(If(..)) => {
                debug!("builtin_bound: bound={}",
                       bound.repr(self.tcx()));
                candidates.vec.push(BuiltinCandidate(bound));
                Ok(())
            }
            Ok(ParameterBuiltin) => { Ok(()) }
            Ok(AmbiguousBuiltin) => {
                debug!("assemble_builtin_bound_candidates: ambiguous builtin");
                Ok(candidates.ambiguous = true)
            }
            Err(e) => { Err(e) }
        }
    }

    fn builtin_bound(&mut self,
                     bound: ty::BuiltinBound,
                     obligation: &TraitObligation<'tcx>)
                     -> Result<BuiltinBoundConditions<'tcx>,SelectionError<'tcx>>
    {
        // Note: these tests operate on types that may contain bound
        // regions. To be proper, we ought to skolemize here, but we
        // forego the skolemization and defer it until the
        // confirmation step.

        let self_ty = self.infcx.shallow_resolve(obligation.predicate.0.self_ty());
        return match self_ty.sty {
            ty::ty_infer(ty::IntVar(_)) |
            ty::ty_infer(ty::FloatVar(_)) |
            ty::ty_uint(_) |
            ty::ty_int(_) |
            ty::ty_bool |
            ty::ty_float(_) |
            ty::ty_bare_fn(..) |
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

            ty::ty_ptr(..) => {     // *const T, *mut T
                match bound {
                    ty::BoundCopy |
                    ty::BoundSized => {
                        Ok(If(Vec::new()))
                    }

                    ty::BoundSync |
                    ty::BoundSend => {
                        // sync and send are not implemented for *const, *mut
                        Err(Unimplemented)
                    }
                }
            }

            ty::ty_trait(ref data) => {
                match bound {
                    ty::BoundSized => {
                        Err(Unimplemented)
                    }
                    ty::BoundCopy | ty::BoundSync | ty::BoundSend => {
                        if data.bounds.builtin_bounds.contains(&bound) {
                            Ok(If(Vec::new()))
                        } else {
                            // Recursively check all supertraits to find out if any further
                            // bounds are required and thus we must fulfill.
                            let principal =
                                data.principal_trait_ref_with_self_ty(self.tcx(),
                                                                      self.tcx().types.err);
                            for tr in util::supertraits(self.tcx(), principal) {
                                let td = ty::lookup_trait_def(self.tcx(), tr.def_id());
                                if td.bounds.builtin_bounds.contains(&bound) {
                                    return Ok(If(Vec::new()))
                                }
                            }

                            Err(Unimplemented)
                        }
                    }
                }
            }

            ty::ty_rptr(_, ty::mt { ty: referent_ty, mutbl }) => {
                // &mut T or &T
                match bound {
                    ty::BoundCopy => {
                        match mutbl {
                            // &mut T is affine and hence never `Copy`
                            ast::MutMutable => {
                                Err(Unimplemented)
                            }

                            // &T is always copyable
                            ast::MutImmutable => {
                                Ok(If(Vec::new()))
                            }
                        }
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

            ty::ty_vec(element_ty, ref len) => {
                // [T, ..n] and [T]
                match bound {
                    ty::BoundCopy => {
                        match *len {
                            Some(_) => {
                                // [T, ..n] is copy iff T is copy
                                Ok(If(vec![element_ty]))
                            }
                            None => {
                                // [T] is unsized and hence affine
                                Err(Unimplemented)
                            }
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
                Ok(If(tys.clone()))
            }

            ty::ty_closure(def_id, _, substs) => {
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

                // Unboxed closures shouldn't be
                // implicitly copyable
                if bound == ty::BoundCopy {
                    return Ok(ParameterBuiltin);
                }

                // Upvars are always local variables or references to
                // local variables, and local variables cannot be
                // unsized, so the closure struct as a whole must be
                // Sized.
                if bound == ty::BoundSized {
                    return Ok(If(Vec::new()));
                }

                match self.closure_typer.closure_upvars(def_id, substs) {
                    Some(upvars) => {
                        Ok(If(upvars.iter().map(|c| c.ty).collect()))
                    }
                    None => {
                        debug!("assemble_builtin_bound_candidates: no upvar types available yet");
                        Ok(AmbiguousBuiltin)
                    }
                }
            }

            ty::ty_struct(def_id, substs) => {
                let types: Vec<Ty> =
                    ty::struct_fields(self.tcx(), def_id, substs).iter()
                                                                 .map(|f| f.mt.ty)
                                                                 .collect();
                nominal(self, bound, def_id, types)
            }

            ty::ty_enum(def_id, substs) => {
                let types: Vec<Ty> =
                    ty::substd_enum_variants(self.tcx(), def_id, substs)
                    .iter()
                    .flat_map(|variant| variant.args.iter())
                    .cloned()
                    .collect();
                nominal(self, bound, def_id, types)
            }

            ty::ty_projection(_) |
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
                debug!("assemble_builtin_bound_candidates: ambiguous builtin");
                Ok(AmbiguousBuiltin)
            }

            ty::ty_open(ty) => {
                // these only crop up in trans, and represent an
                // "opened" unsized/existential type (one that has
                // been dereferenced)
                match bound {
                    ty::BoundCopy |
                    ty::BoundSync |
                    ty::BoundSend => {
                        Ok(If(vec!(ty)))
                    }

                    ty::BoundSized => {
                        Err(Unimplemented)
                    }
                }
            }
            ty::ty_err => {
                Ok(If(Vec::new()))
            }

            ty::ty_infer(ty::FreshTy(_)) |
            ty::ty_infer(ty::FreshIntTy(_)) => {
                self.tcx().sess.bug(
                    &format!(
                        "asked to assemble builtin bounds of unexpected type: {}",
                        self_ty.repr(self.tcx()))[]);
            }
        };

        fn nominal<'cx, 'tcx>(this: &mut SelectionContext<'cx, 'tcx>,
                              bound: ty::BuiltinBound,
                              def_id: ast::DefId,
                              types: Vec<Ty<'tcx>>)
                              -> Result<BuiltinBoundConditions<'tcx>,SelectionError<'tcx>>
        {
            // First check for markers and other nonsense.
            let tcx = this.tcx();
            match bound {
                ty::BoundSend => {
                    if Some(def_id) == tcx.lang_items.managed_bound() {
                        return Err(Unimplemented)
                    }
                }

                ty::BoundCopy => {
                    return Ok(ParameterBuiltin)
                }

                ty::BoundSync => {
                    if
                        Some(def_id) == tcx.lang_items.managed_bound() ||
                        Some(def_id) == tcx.lang_items.unsafe_cell_type()
                    {
                        return Err(Unimplemented)
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
                         obligation: &TraitObligation<'tcx>,
                         candidate: SelectionCandidate<'tcx>)
                         -> Result<Selection<'tcx>,SelectionError<'tcx>>
    {
        debug!("confirm_candidate({}, {})",
               obligation.repr(self.tcx()),
               candidate.repr(self.tcx()));

        match candidate {
            BuiltinCandidate(builtin_bound) => {
                Ok(VtableBuiltin(
                    try!(self.confirm_builtin_candidate(obligation, builtin_bound))))
            }

            PhantomFnCandidate |
            ErrorCandidate => {
                Ok(VtableBuiltin(VtableBuiltinData { nested: VecPerParamSpace::empty() }))
            }

            ParamCandidate(param) => {
                let obligations = self.confirm_param_candidate(obligation, param);
                Ok(VtableParam(obligations))
            }

            ImplCandidate(impl_def_id) => {
                let vtable_impl =
                    try!(self.confirm_impl_candidate(obligation, impl_def_id));
                Ok(VtableImpl(vtable_impl))
            }

            ClosureCandidate(closure_def_id, substs) => {
                try!(self.confirm_closure_candidate(obligation, closure_def_id, &substs));
                Ok(VtableClosure(closure_def_id, substs))
            }

            ObjectCandidate => {
                let data = self.confirm_object_candidate(obligation);
                Ok(VtableObject(data))
            }

            FnPointerCandidate => {
                let fn_type =
                    try!(self.confirm_fn_pointer_candidate(obligation));
                Ok(VtableFnPointer(fn_type))
            }

            ProjectionCandidate => {
                self.confirm_projection_candidate(obligation);
                Ok(VtableParam(Vec::new()))
            }
        }
    }

    fn confirm_projection_candidate(&mut self,
                                    obligation: &TraitObligation<'tcx>)
    {
        let _: Result<(),()> =
            self.infcx.try(|snapshot| {
                let result =
                    self.match_projection_obligation_against_bounds_from_trait(obligation,
                                                                               snapshot);
                assert!(result);
                Ok(())
            });
    }

    fn confirm_param_candidate(&mut self,
                               obligation: &TraitObligation<'tcx>,
                               param: ty::PolyTraitRef<'tcx>)
                               -> Vec<PredicateObligation<'tcx>>
    {
        debug!("confirm_param_candidate({},{})",
               obligation.repr(self.tcx()),
               param.repr(self.tcx()));

        // During evaluation, we already checked that this
        // where-clause trait-ref could be unified with the obligation
        // trait-ref. Repeat that unification now without any
        // transactional boundary; it should not fail.
        match self.match_where_clause_trait_ref(obligation, param.clone()) {
            Ok(obligations) => obligations,
            Err(()) => {
                self.tcx().sess.bug(
                    &format!("Where clause `{}` was applicable to `{}` but now is not",
                             param.repr(self.tcx()),
                             obligation.repr(self.tcx())));
            }
        }
    }

    fn confirm_builtin_candidate(&mut self,
                                 obligation: &TraitObligation<'tcx>,
                                 bound: ty::BuiltinBound)
                                 -> Result<VtableBuiltinData<PredicateObligation<'tcx>>,
                                           SelectionError<'tcx>>
    {
        debug!("confirm_builtin_candidate({})",
               obligation.repr(self.tcx()));

        match try!(self.builtin_bound(bound, obligation)) {
            If(nested) => Ok(self.vtable_builtin_data(obligation, bound, nested)),
            AmbiguousBuiltin | ParameterBuiltin => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    &format!("builtin bound for {} was ambig",
                            obligation.repr(self.tcx()))[]);
            }
        }
    }

    fn vtable_builtin_data(&mut self,
                           obligation: &TraitObligation<'tcx>,
                           bound: ty::BuiltinBound,
                           nested: Vec<Ty<'tcx>>)
                           -> VtableBuiltinData<PredicateObligation<'tcx>>
    {
        let derived_cause = self.derived_cause(obligation, BuiltinDerivedObligation);
        let obligations = nested.iter().map(|&bound_ty| {
            // the obligation might be higher-ranked, e.g. for<'a> &'a
            // int : Copy. In that case, we will wind up with
            // late-bound regions in the `nested` vector. So for each
            // one we instantiate to a skolemized region, do our work
            // to produce something like `&'0 int : Copy`, and then
            // re-bind it. This is a bit of busy-work but preserves
            // the invariant that we only manipulate free regions, not
            // bound ones.
            self.infcx.try(|snapshot| {
                let (skol_ty, skol_map) =
                    self.infcx().skolemize_late_bound_regions(&ty::Binder(bound_ty), snapshot);
                let skol_predicate =
                    util::predicate_for_builtin_bound(
                        self.tcx(),
                        derived_cause.clone(),
                        bound,
                        obligation.recursion_depth + 1,
                        skol_ty);
                match skol_predicate {
                    Ok(skol_predicate) => Ok(self.infcx().plug_leaks(skol_map, snapshot,
                                                                     &skol_predicate)),
                    Err(ErrorReported) => Err(ErrorReported)
                }
            })
        }).collect::<Result<_, _>>();
        let obligations = match obligations {
            Ok(o) => o,
            Err(ErrorReported) => Vec::new()
        };

        let obligations = VecPerParamSpace::new(obligations, Vec::new(), Vec::new());

        debug!("vtable_builtin_data: obligations={}",
               obligations.repr(self.tcx()));

        VtableBuiltinData { nested: obligations }
    }

    fn confirm_impl_candidate(&mut self,
                              obligation: &TraitObligation<'tcx>,
                              impl_def_id: ast::DefId)
                              -> Result<VtableImplData<'tcx, PredicateObligation<'tcx>>,
                                        SelectionError<'tcx>>
    {
        debug!("confirm_impl_candidate({},{})",
               obligation.repr(self.tcx()),
               impl_def_id.repr(self.tcx()));

        // First, create the substitutions by matching the impl again,
        // this time not in a probe.
        self.infcx.try(|snapshot| {
            let (skol_obligation_trait_ref, skol_map) =
                self.infcx().skolemize_late_bound_regions(&obligation.predicate, snapshot);
            let substs =
                self.rematch_impl(impl_def_id, obligation,
                                  snapshot, &skol_map, skol_obligation_trait_ref.trait_ref);
            debug!("confirm_impl_candidate substs={}", substs.repr(self.tcx()));
            Ok(self.vtable_impl(impl_def_id, substs, obligation.cause.clone(),
                                obligation.recursion_depth + 1, skol_map, snapshot))
        })
    }

    fn vtable_impl(&mut self,
                   impl_def_id: ast::DefId,
                   substs: Normalized<'tcx, Substs<'tcx>>,
                   cause: ObligationCause<'tcx>,
                   recursion_depth: uint,
                   skol_map: infer::SkolemizationMap,
                   snapshot: &infer::CombinedSnapshot)
                   -> VtableImplData<'tcx, PredicateObligation<'tcx>>
    {
        debug!("vtable_impl(impl_def_id={}, substs={}, recursion_depth={}, skol_map={})",
               impl_def_id.repr(self.tcx()),
               substs.repr(self.tcx()),
               recursion_depth,
               skol_map.repr(self.tcx()));

        let mut impl_obligations =
            self.impl_obligations(cause,
                                 recursion_depth,
                                 impl_def_id,
                                 &substs.value,
                                 skol_map,
                                 snapshot);

        debug!("vtable_impl: impl_def_id={} impl_obligations={}",
               impl_def_id.repr(self.tcx()),
               impl_obligations.repr(self.tcx()));

        impl_obligations.extend(TypeSpace, substs.obligations.into_iter());

        VtableImplData { impl_def_id: impl_def_id,
                         substs: substs.value,
                         nested: impl_obligations }
    }

    fn confirm_object_candidate(&mut self,
                                obligation: &TraitObligation<'tcx>)
                                -> VtableObjectData<'tcx>
    {
        debug!("confirm_object_candidate({})",
               obligation.repr(self.tcx()));

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        let poly_trait_ref = match self_ty.sty {
            ty::ty_trait(ref data) => {
                data.principal_trait_ref_with_self_ty(self.tcx(), self_ty)
            }
            _ => {
                self.tcx().sess.span_bug(obligation.cause.span,
                                         "object candidate with non-object");
            }
        };

        let obligation_def_id = obligation.predicate.def_id();
        let upcast_trait_ref = match util::upcast(self.tcx(),
                                                  poly_trait_ref.clone(),
                                                  obligation_def_id) {
            Some(r) => r,
            None => {
                self.tcx().sess.span_bug(obligation.cause.span,
                                         &format!("unable to upcast from {} to {}",
                                                  poly_trait_ref.repr(self.tcx()),
                                                  obligation_def_id.repr(self.tcx())));
            }
        };

        match self.match_poly_trait_ref(obligation, upcast_trait_ref) {
            Ok(()) => { }
            Err(()) => {
                self.tcx().sess.span_bug(obligation.cause.span,
                                         "failed to match trait refs");
            }
        }

        VtableObjectData { object_ty: self_ty }
    }

    fn confirm_fn_pointer_candidate(&mut self,
                                    obligation: &TraitObligation<'tcx>)
                                    -> Result<ty::Ty<'tcx>,SelectionError<'tcx>>
    {
        debug!("confirm_fn_pointer_candidate({})",
               obligation.repr(self.tcx()));

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        let sig = ty::ty_fn_sig(self_ty);
        let ty::Binder((trait_ref, _)) =
            util::closure_trait_ref_and_return_type(self.tcx(),
                                                    obligation.predicate.def_id(),
                                                    self_ty,
                                                    sig,
                                                    util::TupleArgumentsFlag::Yes);
        let trait_ref = ty::Binder(trait_ref);

        try!(self.confirm_poly_trait_refs(obligation.cause.clone(),
                                          obligation.predicate.to_poly_trait_ref(),
                                          trait_ref));
        Ok(self_ty)
    }

    fn confirm_closure_candidate(&mut self,
                                 obligation: &TraitObligation<'tcx>,
                                 closure_def_id: ast::DefId,
                                 substs: &Substs<'tcx>)
                                 -> Result<(),SelectionError<'tcx>>
    {
        debug!("confirm_closure_candidate({},{},{})",
               obligation.repr(self.tcx()),
               closure_def_id.repr(self.tcx()),
               substs.repr(self.tcx()));

        let trait_ref = self.closure_trait_ref(obligation,
                                               closure_def_id,
                                               substs);

        debug!("confirm_closure_candidate(closure_def_id={}, trait_ref={})",
               closure_def_id.repr(self.tcx()),
               trait_ref.repr(self.tcx()));

        self.confirm_poly_trait_refs(obligation.cause.clone(),
                                     obligation.predicate.to_poly_trait_ref(),
                                     trait_ref)
    }

    /// In the case of closure types and fn pointers,
    /// we currently treat the input type parameters on the trait as
    /// outputs. This means that when we have a match we have only
    /// considered the self type, so we have to go back and make sure
    /// to relate the argument types too.  This is kind of wrong, but
    /// since we control the full set of impls, also not that wrong,
    /// and it DOES yield better error messages (since we don't report
    /// errors as if there is no applicable impl, but rather report
    /// errors are about mismatched argument types.
    ///
    /// Here is an example. Imagine we have an closure expression
    /// and we desugared it so that the type of the expression is
    /// `Closure`, and `Closure` expects an int as argument. Then it
    /// is "as if" the compiler generated this impl:
    ///
    ///     impl Fn(int) for Closure { ... }
    ///
    /// Now imagine our obligation is `Fn(uint) for Closure`. So far
    /// we have matched the self-type `Closure`. At this point we'll
    /// compare the `int` to `uint` and generate an error.
    ///
    /// Note that this checking occurs *after* the impl has selected,
    /// because these output type parameters should not affect the
    /// selection of the impl. Therefore, if there is a mismatch, we
    /// report an error to the user.
    fn confirm_poly_trait_refs(&mut self,
                               obligation_cause: ObligationCause,
                               obligation_trait_ref: ty::PolyTraitRef<'tcx>,
                               expected_trait_ref: ty::PolyTraitRef<'tcx>)
                               -> Result<(), SelectionError<'tcx>>
    {
        let origin = infer::RelateOutputImplTypes(obligation_cause.span);

        let obligation_trait_ref = obligation_trait_ref.clone();
        match self.infcx.sub_poly_trait_refs(false,
                                             origin,
                                             expected_trait_ref.clone(),
                                             obligation_trait_ref.clone()) {
            Ok(()) => Ok(()),
            Err(e) => Err(OutputTypeParameterMismatch(expected_trait_ref, obligation_trait_ref, e))
        }
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
                    obligation: &TraitObligation<'tcx>,
                    snapshot: &infer::CombinedSnapshot,
                    skol_map: &infer::SkolemizationMap,
                    skol_obligation_trait_ref: Rc<ty::TraitRef<'tcx>>)
                    -> Normalized<'tcx, Substs<'tcx>>
    {
        match self.match_impl(impl_def_id, obligation, snapshot,
                              skol_map, skol_obligation_trait_ref) {
            Ok(substs) => {
                substs
            }
            Err(()) => {
                self.tcx().sess.bug(
                    &format!("Impl {} was matchable against {} but now is not",
                            impl_def_id.repr(self.tcx()),
                            obligation.repr(self.tcx()))[]);
            }
        }
    }

    fn match_impl(&mut self,
                  impl_def_id: ast::DefId,
                  obligation: &TraitObligation<'tcx>,
                  snapshot: &infer::CombinedSnapshot,
                  skol_map: &infer::SkolemizationMap,
                  skol_obligation_trait_ref: Rc<ty::TraitRef<'tcx>>)
                  -> Result<Normalized<'tcx, Substs<'tcx>>, ()>
    {
        let impl_trait_ref = ty::impl_trait_ref(self.tcx(), impl_def_id).unwrap();

        // Before we create the substitutions and everything, first
        // consider a "quick reject". This avoids creating more types
        // and so forth that we need to.
        if self.fast_reject_trait_refs(obligation, &*impl_trait_ref) {
            return Err(());
        }

        let impl_substs = util::fresh_substs_for_impl(self.infcx,
                                                      obligation.cause.span,
                                                      impl_def_id);

        let impl_trait_ref = impl_trait_ref.subst(self.tcx(),
                                                  &impl_substs);

        let impl_trait_ref =
            project::normalize_with_depth(self,
                                          obligation.cause.clone(),
                                          obligation.recursion_depth + 1,
                                          &impl_trait_ref);

        debug!("match_impl(impl_def_id={}, obligation={}, \
               impl_trait_ref={}, skol_obligation_trait_ref={})",
               impl_def_id.repr(self.tcx()),
               obligation.repr(self.tcx()),
               impl_trait_ref.repr(self.tcx()),
               skol_obligation_trait_ref.repr(self.tcx()));

        let origin = infer::RelateOutputImplTypes(obligation.cause.span);
        match self.infcx.sub_trait_refs(false,
                                        origin,
                                        impl_trait_ref.value.clone(),
                                        skol_obligation_trait_ref) {
            Ok(()) => { }
            Err(e) => {
                debug!("match_impl: failed sub_trait_refs due to `{}`",
                       ty::type_err_to_str(self.tcx(), &e));
                return Err(());
            }
        }

        match self.infcx.leak_check(skol_map, snapshot) {
            Ok(()) => { }
            Err(e) => {
                debug!("match_impl: failed leak check due to `{}`",
                       ty::type_err_to_str(self.tcx(), &e));
                return Err(());
            }
        }

        debug!("match_impl: success impl_substs={}", impl_substs.repr(self.tcx()));
        Ok(Normalized { value: impl_substs,
                        obligations: impl_trait_ref.obligations })
    }

    fn fast_reject_trait_refs(&mut self,
                              obligation: &TraitObligation,
                              impl_trait_ref: &ty::TraitRef)
                              -> bool
    {
        // We can avoid creating type variables and doing the full
        // substitution if we find that any of the input types, when
        // simplified, do not match.

        obligation.predicate.0.input_types().iter()
            .zip(impl_trait_ref.input_types().iter())
            .any(|(&obligation_ty, &impl_ty)| {
                let simplified_obligation_ty =
                    fast_reject::simplify_type(self.tcx(), obligation_ty, true);
                let simplified_impl_ty =
                    fast_reject::simplify_type(self.tcx(), impl_ty, false);

                simplified_obligation_ty.is_some() &&
                    simplified_impl_ty.is_some() &&
                    simplified_obligation_ty != simplified_impl_ty
            })
    }

    /// Normalize `where_clause_trait_ref` and try to match it against
    /// `obligation`.  If successful, return any predicates that
    /// result from the normalization. Normalization is necessary
    /// because where-clauses are stored in the parameter environment
    /// unnormalized.
    fn match_where_clause_trait_ref(&mut self,
                                    obligation: &TraitObligation<'tcx>,
                                    where_clause_trait_ref: ty::PolyTraitRef<'tcx>)
                                    -> Result<Vec<PredicateObligation<'tcx>>,()>
    {
        let () =
            try!(self.match_poly_trait_ref(obligation, where_clause_trait_ref));

        Ok(Vec::new())
    }

    /// Returns `Ok` if `poly_trait_ref` being true implies that the
    /// obligation is satisfied.
    fn match_poly_trait_ref(&mut self,
                            obligation: &TraitObligation<'tcx>,
                            poly_trait_ref: ty::PolyTraitRef<'tcx>)
                            -> Result<(),()>
    {
        debug!("match_poly_trait_ref: obligation={} poly_trait_ref={}",
               obligation.repr(self.tcx()),
               poly_trait_ref.repr(self.tcx()));

        let origin = infer::RelateOutputImplTypes(obligation.cause.span);
        match self.infcx.sub_poly_trait_refs(false,
                                             origin,
                                             poly_trait_ref,
                                             obligation.predicate.to_poly_trait_ref()) {
            Ok(()) => Ok(()),
            Err(_) => Err(()),
        }
    }

    /// Determines whether the self type declared against
    /// `impl_def_id` matches `obligation_self_ty`. If successful,
    /// returns the substitutions used to make them match. See
    /// `match_impl()`. For example, if `impl_def_id` is declared
    /// as:
    ///
    ///    impl<T:Copy> Foo for ~T { ... }
    ///
    /// and `obligation_self_ty` is `int`, we'd back an `Err(_)`
    /// result. But if `obligation_self_ty` were `~int`, we'd get
    /// back `Ok(T=int)`.
    fn match_inherent_impl(&mut self,
                           impl_def_id: ast::DefId,
                           obligation_cause: &ObligationCause,
                           obligation_self_ty: Ty<'tcx>)
                           -> Result<Substs<'tcx>,()>
    {
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
                        cause: &ObligationCause,

                        // The self type provided by the impl/caller-obligation:
                        provided_self_ty: Ty<'tcx>,

                        // The self type the obligation is for:
                        required_self_ty: Ty<'tcx>)
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
    // Miscellany

    fn push_stack<'o,'s:'o>(&mut self,
                            previous_stack: Option<&'s TraitObligationStack<'s, 'tcx>>,
                            obligation: &'o TraitObligation<'tcx>)
                            -> TraitObligationStack<'o, 'tcx>
    {
        let fresh_trait_ref =
            obligation.predicate.to_poly_trait_ref().fold_with(&mut self.freshener);

        TraitObligationStack {
            obligation: obligation,
            fresh_trait_ref: fresh_trait_ref,
            previous: previous_stack.map(|p| p), // FIXME variance
        }
    }

    /// Returns set of all impls for a given trait.
    fn all_impls(&self, trait_def_id: ast::DefId) -> Vec<ast::DefId> {
        ty::populate_implementations_for_trait_if_necessary(self.tcx(), trait_def_id);

        match self.tcx().trait_impls.borrow().get(&trait_def_id) {
            None => Vec::new(),
            Some(impls) => impls.borrow().clone()
        }
    }

    fn closure_trait_ref(&self,
                         obligation: &TraitObligation<'tcx>,
                         closure_def_id: ast::DefId,
                         substs: &Substs<'tcx>)
                         -> ty::PolyTraitRef<'tcx>
    {
        let closure_type = self.closure_typer.closure_type(closure_def_id, substs);
        let ty::Binder((trait_ref, _)) =
            util::closure_trait_ref_and_return_type(self.tcx(),
                                                    obligation.predicate.def_id(),
                                                    obligation.predicate.0.self_ty(), // (1)
                                                    &closure_type.sig,
                                                    util::TupleArgumentsFlag::No);

        // (1) Feels icky to skip the binder here, but OTOH we know
        // that the self-type is an unboxed closure type and hence is
        // in fact unparameterized (or at least does not reference any
        // regions bound in the obligation). Still probably some
        // refactoring could make this nicer.

        ty::Binder(trait_ref)
    }

    fn impl_obligations(&mut self,
                        cause: ObligationCause<'tcx>,
                        recursion_depth: uint,
                        impl_def_id: ast::DefId,
                        impl_substs: &Substs<'tcx>,
                        skol_map: infer::SkolemizationMap,
                        snapshot: &infer::CombinedSnapshot)
                        -> VecPerParamSpace<PredicateObligation<'tcx>>
    {
        let impl_bounds = ty::lookup_predicates(self.tcx(), impl_def_id);
        let bounds = impl_bounds.instantiate(self.tcx(), impl_substs);
        let normalized_bounds =
            project::normalize_with_depth(self, cause.clone(), recursion_depth, &bounds);
        let normalized_bounds =
            self.infcx().plug_leaks(skol_map, snapshot, &normalized_bounds);
        let mut impl_obligations =
            util::predicates_for_generics(self.tcx(),
                                          cause,
                                          recursion_depth,
                                          &normalized_bounds.value);
        impl_obligations.extend(TypeSpace, normalized_bounds.obligations.into_iter());
        impl_obligations
    }

    #[allow(unused_comparisons)]
    fn derived_cause(&self,
                     obligation: &TraitObligation<'tcx>,
                     variant: fn(DerivedObligationCause<'tcx>) -> ObligationCauseCode<'tcx>)
                     -> ObligationCause<'tcx>
    {
        /*!
         * Creates a cause for obligations that are derived from
         * `obligation` by a recursive search (e.g., for a builtin
         * bound, or eventually a `impl Foo for ..`). If `obligation`
         * is itself a derived obligation, this is just a clone, but
         * otherwise we create a "derived obligation" cause so as to
         * keep track of the original root obligation for error
         * reporting.
         */

        // NOTE(flaper87): As of now, it keeps track of the whole error
        // chain. Ideally, we should have a way to configure this either
        // by using -Z verbose or just a CLI argument.
        if obligation.recursion_depth >= 0 {
            let derived_cause = DerivedObligationCause {
                parent_trait_ref: obligation.predicate.to_poly_trait_ref(),
                parent_code: Rc::new(obligation.cause.code.clone()),
            };
            ObligationCause::new(obligation.cause.span,
                                 obligation.cause.body_id,
                                 variant(derived_cause))
        } else {
            obligation.cause.clone()
        }
    }
}

impl<'tcx> Repr<'tcx> for SelectionCandidate<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            PhantomFnCandidate => format!("PhantomFnCandidate"),
            ErrorCandidate => format!("ErrorCandidate"),
            BuiltinCandidate(b) => format!("BuiltinCandidate({:?})", b),
            ParamCandidate(ref a) => format!("ParamCandidate({})", a.repr(tcx)),
            ImplCandidate(a) => format!("ImplCandidate({})", a.repr(tcx)),
            ProjectionCandidate => format!("ProjectionCandidate"),
            FnPointerCandidate => format!("FnPointerCandidate"),
            ObjectCandidate => {
                format!("ObjectCandidate")
            }
            ClosureCandidate(c, ref s) => {
                format!("ClosureCandidate({:?},{})", c, s.repr(tcx))
            }
        }
    }
}

impl<'tcx> SelectionCache<'tcx> {
    pub fn new() -> SelectionCache<'tcx> {
        SelectionCache {
            hashmap: RefCell::new(HashMap::new())
        }
    }
}

impl<'o, 'tcx> TraitObligationStack<'o, 'tcx> {
    fn iter(&self) -> Option<&TraitObligationStack<'o, 'tcx>> {
        Some(self)
    }
}

impl<'o, 'tcx> Iterator for Option<&'o TraitObligationStack<'o, 'tcx>> {
    type Item = &'o TraitObligationStack<'o,'tcx>;

    fn next(&mut self) -> Option<&'o TraitObligationStack<'o, 'tcx>> {
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

impl<'o, 'tcx> Repr<'tcx> for TraitObligationStack<'o, 'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("TraitObligationStack({})",
                self.obligation.repr(tcx))
    }
}

impl<'tcx> EvaluationResult<'tcx> {
    fn may_apply(&self) -> bool {
        match *self {
            EvaluatedToOk |
            EvaluatedToAmbig |
            EvaluatedToErr(Overflow) |
            EvaluatedToErr(OutputTypeParameterMismatch(..)) => {
                true
            }
            EvaluatedToErr(Unimplemented) => {
                false
            }
        }
    }
}

impl MethodMatchResult {
    pub fn may_apply(&self) -> bool {
        match *self {
            MethodMatched(_) => true,
            MethodAmbiguous(_) => true,
            MethodDidNotMatch => false,
        }
    }
}
