// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See `README.md` for high-level documentation

pub use self::MethodMatchResult::*;
pub use self::MethodMatchedData::*;
use self::SelectionCandidate::*;
use self::BuiltinBoundConditions::*;
use self::EvaluationResult::*;

use super::coherence;
use super::DerivedObligationCause;
use super::project;
use super::project::{normalize_with_depth, Normalized};
use super::{PredicateObligation, TraitObligation, ObligationCause};
use super::report_overflow_error;
use super::{ObligationCauseCode, BuiltinDerivedObligation, ImplDerivedObligation};
use super::{SelectionError, Unimplemented, OutputTypeParameterMismatch};
use super::{ObjectCastObligation, Obligation};
use super::TraitNotObjectSafe;
use super::Selection;
use super::SelectionResult;
use super::{VtableBuiltin, VtableImpl, VtableParam, VtableClosure,
            VtableFnPointer, VtableObject, VtableDefaultImpl};
use super::{VtableImplData, VtableObjectData, VtableBuiltinData,
            VtableClosureData, VtableDefaultImplData};
use super::object_safety;
use super::util;

use middle::def_id::DefId;
use middle::infer;
use middle::infer::{InferCtxt, TypeFreshener, TypeOrigin};
use middle::subst::{Subst, Substs, TypeSpace};
use middle::ty::{self, ToPredicate, ToPolyTraitRef, Ty, TypeFoldable};
use middle::ty::fast_reject;
use middle::ty::relate::TypeRelation;

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use syntax::abi;
use rustc_front::hir;
use util::common::ErrorReported;
use util::nodemap::FnvHashMap;

pub struct SelectionContext<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,

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

    previous: TraitObligationStackList<'prev, 'tcx>,
}

#[derive(Clone)]
pub struct SelectionCache<'tcx> {
    hashmap: RefCell<FnvHashMap<ty::TraitRef<'tcx>,
                                SelectionResult<'tcx, SelectionCandidate<'tcx>>>>,
}

pub enum MethodMatchResult {
    MethodMatched(MethodMatchedData),
    MethodAmbiguous(/* list of impls that could apply */ Vec<DefId>),
    MethodDidNotMatch,
}

#[derive(Copy, Clone, Debug)]
pub enum MethodMatchedData {
    // In the case of a precise match, we don't really need to store
    // how the match was found. So don't.
    PreciseMethodMatch,

    // In the case of a coercion, we need to know the precise impl so
    // that we can determine the type to which things were coerced.
    CoerciveMethodMatch(/* impl we matched */ DefId)
}

/// The selection process begins by considering all impls, where
/// clauses, and so forth that might resolve an obligation.  Sometimes
/// we'll be able to say definitively that (e.g.) an impl does not
/// apply to the obligation: perhaps it is defined for `usize` but the
/// obligation is for `int`. In that case, we drop the impl out of the
/// list.  But the other cases are considered *candidates*.
///
/// For selection to succeed, there must be exactly one matching
/// candidate. If the obligation is fully known, this is guaranteed
/// by coherence. However, if the obligation contains type parameters
/// or variables, there may be multiple such impls.
///
/// It is not a real problem if multiple matching impls exist because
/// of type variables - it just means the obligation isn't sufficiently
/// elaborated. In that case we report an ambiguity, and the caller can
/// try again after more type information has been gathered or report a
/// "type annotations required" error.
///
/// However, with type parameters, this can be a real problem - type
/// parameters don't unify with regular types, but they *can* unify
/// with variables from blanket impls, and (unless we know its bounds
/// will always be satisfied) picking the blanket impl will be wrong
/// for at least *some* substitutions. To make this concrete, if we have
///
///    trait AsDebug { type Out : fmt::Debug; fn debug(self) -> Self::Out; }
///    impl<T: fmt::Debug> AsDebug for T {
///        type Out = T;
///        fn debug(self) -> fmt::Debug { self }
///    }
///    fn foo<T: AsDebug>(t: T) { println!("{:?}", <T as AsDebug>::debug(t)); }
///
/// we can't just use the impl to resolve the <T as AsDebug> obligation
/// - a type from another crate (that doesn't implement fmt::Debug) could
/// implement AsDebug.
///
/// Because where-clauses match the type exactly, multiple clauses can
/// only match if there are unresolved variables, and we can mostly just
/// report this ambiguity in that case. This is still a problem - we can't
/// *do anything* with ambiguities that involve only regions. This is issue
/// #21974.
///
/// If a single where-clause matches and there are no inference
/// variables left, then it definitely matches and we can just select
/// it.
///
/// In fact, we even select the where-clause when the obligation contains
/// inference variables. The can lead to inference making "leaps of logic",
/// for example in this situation:
///
///    pub trait Foo<T> { fn foo(&self) -> T; }
///    impl<T> Foo<()> for T { fn foo(&self) { } }
///    impl Foo<bool> for bool { fn foo(&self) -> bool { *self } }
///
///    pub fn foo<T>(t: T) where T: Foo<bool> {
///       println!("{:?}", <T as Foo<_>>::foo(&t));
///    }
///    fn main() { foo(false); }
///
/// Here the obligation <T as Foo<$0>> can be matched by both the blanket
/// impl and the where-clause. We select the where-clause and unify $0=bool,
/// so the program prints "false". However, if the where-clause is omitted,
/// the blanket impl is selected, we unify $0=(), and the program prints
/// "()".
///
/// Exactly the same issues apply to projection and object candidates, except
/// that we can have both a projection candidate and a where-clause candidate
/// for the same obligation. In that case either would do (except that
/// different "leaps of logic" would occur if inference variables are
/// present), and we just pick the where-clause. This is, for example,
/// required for associated types to work in default impls, as the bounds
/// are visible both as projection bounds and as where-clauses from the
/// parameter environment.
#[derive(PartialEq,Eq,Debug,Clone)]
enum SelectionCandidate<'tcx> {
    BuiltinCandidate(ty::BuiltinBound),
    ParamCandidate(ty::PolyTraitRef<'tcx>),
    ImplCandidate(DefId),
    DefaultImplCandidate(DefId),
    DefaultImplObjectCandidate(DefId),

    /// This is a trait matching with a projected type as `Self`, and
    /// we found an applicable bound in the trait definition.
    ProjectionCandidate,

    /// Implementation of a `Fn`-family trait by one of the
    /// anonymous types generated for a `||` expression.
    ClosureCandidate(/* closure */ DefId, &'tcx ty::ClosureSubsts<'tcx>),

    /// Implementation of a `Fn`-family trait by one of the anonymous
    /// types generated for a fn pointer type (e.g., `fn(int)->int`)
    FnPointerCandidate,

    ObjectCandidate,

    BuiltinObjectCandidate,

    BuiltinUnsizeCandidate,
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
    If(ty::Binder<Vec<Ty<'tcx>>>),
    ParameterBuiltin,
    AmbiguousBuiltin
}

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
/// The result of trait evaluation. The order is important
/// here as the evaluation of a list is the maximum of the
/// evaluations.
enum EvaluationResult {
    /// Evaluation successful
    EvaluatedToOk,
    /// Evaluation failed because of recursion - treated as ambiguous
    EvaluatedToUnknown,
    /// Evaluation is known to be ambiguous
    EvaluatedToAmbig,
    /// Evaluation failed
    EvaluatedToErr,
}

#[derive(Clone)]
pub struct EvaluationCache<'tcx> {
    hashmap: RefCell<FnvHashMap<ty::PolyTraitRef<'tcx>, EvaluationResult>>
}

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    pub fn new(infcx: &'cx InferCtxt<'cx, 'tcx>)
               -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
            freshener: infcx.freshener(),
            intercrate: false,
        }
    }

    pub fn intercrate(infcx: &'cx InferCtxt<'cx, 'tcx>)
                      -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
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
        self.infcx.param_env()
    }

    pub fn closure_typer(&self) -> &'cx InferCtxt<'cx, 'tcx> {
        self.infcx
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
        debug!("select({:?})", obligation);
        assert!(!obligation.predicate.has_escaping_regions());

        let dep_node = obligation.dep_node(self.tcx());
        let _task = self.tcx().dep_graph.in_task(dep_node);

        let stack = self.push_stack(TraitObligationStackList::empty(), obligation);
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
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let (closure_def_id, substs) = match self_ty.sty {
            ty::TyClosure(id, ref substs) => (id, substs),
            _ => { return; }
        };
        assert!(!substs.has_escaping_regions());

        // It is OK to call the unnormalized variant here - this is only
        // reached for TyClosure: Fn inputs where the closure kind is
        // still unknown, which should only occur in typeck where the
        // closure type is already normalized.
        let closure_trait_ref = self.closure_trait_ref_unnormalized(obligation,
                                                                    closure_def_id,
                                                                    substs);

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
        debug!("evaluate_obligation({:?})",
               obligation);

        self.infcx.probe(|_| {
            self.evaluate_predicate_recursively(TraitObligationStackList::empty(), obligation)
                .may_apply()
        })
    }

    /// Evaluates whether the obligation `obligation` can be satisfied,
    /// and returns `false` if not certain. However, this is not entirely
    /// accurate if inference variables are involved.
    pub fn evaluate_obligation_conservatively(&mut self,
                                              obligation: &PredicateObligation<'tcx>)
                                              -> bool
    {
        debug!("evaluate_obligation_conservatively({:?})",
               obligation);

        self.infcx.probe(|_| {
            self.evaluate_predicate_recursively(TraitObligationStackList::empty(), obligation)
                == EvaluatedToOk
        })
    }

    /// Evaluates the predicates in `predicates` recursively. Note that
    /// this applies projections in the predicates, and therefore
    /// is run within an inference probe.
    fn evaluate_predicates_recursively<'a,'o,I>(&mut self,
                                                stack: TraitObligationStackList<'o, 'tcx>,
                                                predicates: I)
                                                -> EvaluationResult
        where I : Iterator<Item=&'a PredicateObligation<'tcx>>, 'tcx:'a
    {
        let mut result = EvaluatedToOk;
        for obligation in predicates {
            let eval = self.evaluate_predicate_recursively(stack, obligation);
            debug!("evaluate_predicate_recursively({:?}) = {:?}",
                   obligation, eval);
            match eval {
                EvaluatedToErr => { return EvaluatedToErr; }
                EvaluatedToAmbig => { result = EvaluatedToAmbig; }
                EvaluatedToUnknown => {
                    if result < EvaluatedToUnknown {
                        result = EvaluatedToUnknown;
                    }
                }
                EvaluatedToOk => { }
            }
        }
        result
    }

    fn evaluate_predicate_recursively<'o>(&mut self,
                                          previous_stack: TraitObligationStackList<'o, 'tcx>,
                                          obligation: &PredicateObligation<'tcx>)
                                           -> EvaluationResult
    {
        debug!("evaluate_predicate_recursively({:?})",
               obligation);

        // Check the cache from the tcx of predicates that we know
        // have been proven elsewhere. This cache only contains
        // predicates that are global in scope and hence unaffected by
        // the current environment.
        if self.tcx().fulfilled_predicates.borrow().is_duplicate(&obligation.predicate) {
            return EvaluatedToOk;
        }

        match obligation.predicate {
            ty::Predicate::Trait(ref t) => {
                assert!(!t.has_escaping_regions());
                let obligation = obligation.with(t.clone());
                self.evaluate_obligation_recursively(previous_stack, &obligation)
            }

            ty::Predicate::Equate(ref p) => {
                // does this code ever run?
                match self.infcx.equality_predicate(obligation.cause.span, p) {
                    Ok(()) => EvaluatedToOk,
                    Err(_) => EvaluatedToErr
                }
            }

            ty::Predicate::WellFormed(ty) => {
                match ty::wf::obligations(self.infcx, obligation.cause.body_id,
                                          ty, obligation.cause.span) {
                    Some(obligations) =>
                        self.evaluate_predicates_recursively(previous_stack, obligations.iter()),
                    None =>
                        EvaluatedToAmbig,
                }
            }

            ty::Predicate::TypeOutlives(..) | ty::Predicate::RegionOutlives(..) => {
                // we do not consider region relationships when
                // evaluating trait matches
                EvaluatedToOk
            }

            ty::Predicate::ObjectSafe(trait_def_id) => {
                if object_safety::is_object_safe(self.tcx(), trait_def_id) {
                    EvaluatedToOk
                } else {
                    EvaluatedToErr
                }
            }

            ty::Predicate::Projection(ref data) => {
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
                        EvaluatedToErr
                    }
                }
            }
        }
    }

    fn evaluate_obligation_recursively<'o>(&mut self,
                                           previous_stack: TraitObligationStackList<'o, 'tcx>,
                                           obligation: &TraitObligation<'tcx>)
                                           -> EvaluationResult
    {
        debug!("evaluate_obligation_recursively({:?})",
               obligation);

        let stack = self.push_stack(previous_stack, obligation);
        let fresh_trait_ref = stack.fresh_trait_ref;
        if let Some(result) = self.check_evaluation_cache(fresh_trait_ref) {
            debug!("CACHE HIT: EVAL({:?})={:?}",
                   fresh_trait_ref,
                   result);
            return result;
        }

        let result = self.evaluate_stack(&stack);

        debug!("CACHE MISS: EVAL({:?})={:?}",
               fresh_trait_ref,
               result);
        self.insert_evaluation_cache(fresh_trait_ref, result);

        result
    }

    fn evaluate_stack<'o>(&mut self,
                          stack: &TraitObligationStack<'o, 'tcx>)
                          -> EvaluationResult
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
        let unbound_input_types = input_types.iter().any(|ty| ty.is_fresh());
        if unbound_input_types && self.intercrate {
            debug!("evaluate_stack({:?}) --> unbound argument, intercrate -->  ambiguous",
                   stack.fresh_trait_ref);
            return EvaluatedToAmbig;
        }
        if unbound_input_types &&
              stack.iter().skip(1).any(
                  |prev| self.match_fresh_trait_refs(&stack.fresh_trait_ref,
                                                     &prev.fresh_trait_ref))
        {
            debug!("evaluate_stack({:?}) --> unbound argument, recursive --> giving up",
                   stack.fresh_trait_ref);
            return EvaluatedToUnknown;
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
            debug!("evaluate_stack({:?}) --> recursive",
                   stack.fresh_trait_ref);
            return EvaluatedToOk;
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.evaluate_candidate(stack, &c),
            Ok(None) => EvaluatedToAmbig,
            Err(..) => EvaluatedToErr
        }
    }

    /// Further evaluate `candidate` to decide whether all type parameters match and whether nested
    /// obligations are met. Returns true if `candidate` remains viable after this further
    /// scrutiny.
    fn evaluate_candidate<'o>(&mut self,
                              stack: &TraitObligationStack<'o, 'tcx>,
                              candidate: &SelectionCandidate<'tcx>)
                              -> EvaluationResult
    {
        debug!("evaluate_candidate: depth={} candidate={:?}",
               stack.obligation.recursion_depth, candidate);
        let result = self.infcx.probe(|_| {
            let candidate = (*candidate).clone();
            match self.confirm_candidate(stack.obligation, candidate) {
                Ok(selection) => {
                    self.evaluate_predicates_recursively(
                        stack.list(),
                        selection.nested_obligations().iter())
                }
                Err(..) => EvaluatedToErr
            }
        });
        debug!("evaluate_candidate: depth={} result={:?}",
               stack.obligation.recursion_depth, result);
        result
    }

    fn pick_evaluation_cache(&self) -> &EvaluationCache<'tcx> {
        // see comment in `pick_candidate_cache`
        if self.intercrate ||
            !self.param_env().caller_bounds.is_empty()
        {
            &self.param_env().evaluation_cache
        } else
        {
            &self.tcx().evaluation_cache
        }
    }

    fn check_evaluation_cache(&self, trait_ref: ty::PolyTraitRef<'tcx>)
                              -> Option<EvaluationResult>
    {
        let cache = self.pick_evaluation_cache();
        cache.hashmap.borrow().get(&trait_ref).cloned()
    }

    fn insert_evaluation_cache(&mut self,
                               trait_ref: ty::PolyTraitRef<'tcx>,
                               result: EvaluationResult)
    {
        // Avoid caching results that depend on more than just the trait-ref:
        // The stack can create EvaluatedToUnknown, and closure signatures
        // being yet uninferred can create "spurious" EvaluatedToAmbig
        // and EvaluatedToOk.
        if result == EvaluatedToUnknown ||
            ((result == EvaluatedToAmbig || result == EvaluatedToOk)
             && trait_ref.has_closure_types())
        {
            return;
        }

        let cache = self.pick_evaluation_cache();
        cache.hashmap.borrow_mut().insert(trait_ref, result);
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY
    //
    // The selection process begins by examining all in-scope impls,
    // caller obligations, and so forth and assembling a list of
    // candidates. See `README.md` and the `Candidate` type for more
    // details.

    fn candidate_from_obligation<'o>(&mut self,
                                     stack: &TraitObligationStack<'o, 'tcx>)
                                     -> SelectionResult<'tcx, SelectionCandidate<'tcx>>
    {
        // Watch out for overflow. This intentionally bypasses (and does
        // not update) the cache.
        let recursion_limit = self.infcx.tcx.sess.recursion_limit.get();
        if stack.obligation.recursion_depth >= recursion_limit {
            report_overflow_error(self.infcx(), &stack.obligation);
        }

        // Check the cache. Note that we skolemize the trait-ref
        // separately rather than using `stack.fresh_trait_ref` -- this
        // is because we want the unbound variables to be replaced
        // with fresh skolemized types starting from index 0.
        let cache_fresh_trait_pred =
            self.infcx.freshen(stack.obligation.predicate.clone());
        debug!("candidate_from_obligation(cache_fresh_trait_pred={:?}, obligation={:?})",
               cache_fresh_trait_pred,
               stack);
        assert!(!stack.obligation.predicate.has_escaping_regions());

        match self.check_candidate_cache(&cache_fresh_trait_pred) {
            Some(c) => {
                debug!("CACHE HIT: SELECT({:?})={:?}",
                       cache_fresh_trait_pred,
                       c);
                return c;
            }
            None => { }
        }

        // If no match, compute result and insert into cache.
        let candidate = self.candidate_from_obligation_no_cache(stack);

        if self.should_update_candidate_cache(&cache_fresh_trait_pred, &candidate) {
            debug!("CACHE MISS: SELECT({:?})={:?}",
                   cache_fresh_trait_pred, candidate);
            self.insert_candidate_cache(cache_fresh_trait_pred, candidate.clone());
        }

        candidate
    }

    fn candidate_from_obligation_no_cache<'o>(&mut self,
                                              stack: &TraitObligationStack<'o, 'tcx>)
                                              -> SelectionResult<'tcx, SelectionCandidate<'tcx>>
    {
        if stack.obligation.predicate.references_error() {
            // If we encounter a `TyError`, we generally prefer the
            // most "optimistic" result in response -- that is, the
            // one least likely to report downstream errors. But
            // because this routine is shared by coherence and by
            // trait selection, there isn't an obvious "right" choice
            // here in that respect, so we opt to just return
            // ambiguity and let the upstream clients sort it out.
            return Ok(None);
        }

        if !self.is_knowable(stack) {
            debug!("intercrate not knowable");
            return Ok(None);
        }

        let candidate_set = try!(self.assemble_candidates(stack));

        if candidate_set.ambiguous {
            debug!("candidate set contains ambig");
            return Ok(None);
        }

        let mut candidates = candidate_set.vec;

        debug!("assembled {} candidates for {:?}: {:?}",
               candidates.len(),
               stack,
               candidates);

        // At this point, we know that each of the entries in the
        // candidate set is *individually* applicable. Now we have to
        // figure out if they contain mutual incompatibilities. This
        // frequently arises if we have an unconstrained input type --
        // for example, we are looking for $0:Eq where $0 is some
        // unconstrained type variable. In that case, we'll get a
        // candidate which assumes $0 == int, one that assumes $0 ==
        // usize, etc. This spells an ambiguity.

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
            candidates.retain(|c| self.evaluate_candidate(stack, c).may_apply())
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
                    debug!("Dropping candidate #{}/{}: {:?}",
                           i, candidates.len(), candidates[i]);
                    candidates.swap_remove(i);
                } else {
                    debug!("Retaining candidate #{}/{}: {:?}",
                           i, candidates.len(), candidates[i]);
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
        if candidates.is_empty() {
            return Err(Unimplemented);
        }

        // Just one candidate left.
        let candidate = candidates.pop().unwrap();

        match candidate {
            ImplCandidate(def_id) => {
                match self.tcx().trait_impl_polarity(def_id) {
                    Some(hir::ImplPolarity::Negative) => return Err(Unimplemented),
                    _ => {}
                }
            }
            _ => {}
        }

        Ok(Some(candidate))
    }

    fn is_knowable<'o>(&mut self,
                       stack: &TraitObligationStack<'o, 'tcx>)
                       -> bool
    {
        debug!("is_knowable(intercrate={})", self.intercrate);

        if !self.intercrate {
            return true;
        }

        let obligation = &stack.obligation;
        let predicate = self.infcx().resolve_type_vars_if_possible(&obligation.predicate);

        // ok to skip binder because of the nature of the
        // trait-ref-is-knowable check, which does not care about
        // bound regions
        let trait_ref = &predicate.skip_binder().trait_ref;

        coherence::trait_ref_is_knowable(self.tcx(), trait_ref)
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
                cache_fresh_trait_pred.0.trait_ref.substs.types.has_infer_types()
            }
        }
    }

    fn assemble_candidates<'o>(&mut self,
                               stack: &TraitObligationStack<'o, 'tcx>)
                               -> Result<SelectionCandidateSet<'tcx>, SelectionError<'tcx>>
    {
        let TraitObligationStack { obligation, .. } = *stack;
        let ref obligation = Obligation {
            cause: obligation.cause.clone(),
            recursion_depth: obligation.recursion_depth,
            predicate: self.infcx().resolve_type_vars_if_possible(&obligation.predicate)
        };

        if obligation.predicate.skip_binder().self_ty().is_ty_var() {
            // FIXME(#20297): Self is a type variable (e.g. `_: AsRef<str>`).
            //
            // This is somewhat problematic, as the current scheme can't really
            // handle it turning to be a projection. This does end up as truly
            // ambiguous in most cases anyway.
            //
            // Until this is fixed, take the fast path out - this also improves
            // performance by preventing assemble_candidates_from_impls from
            // matching every impl for this trait.
            return Ok(SelectionCandidateSet { vec: vec![], ambiguous: true });
        }

        let mut candidates = SelectionCandidateSet {
            vec: Vec::new(),
            ambiguous: false
        };

        // Other bounds. Consider both in-scope bounds from fn decl
        // and applicable impls. There is a certain set of precedence rules here.

        match self.tcx().lang_items.to_builtin_kind(obligation.predicate.def_id()) {
            Some(ty::BoundCopy) => {
                debug!("obligation self ty is {:?}",
                       obligation.predicate.0.self_ty());

                // User-defined copy impls are permitted, but only for
                // structs and enums.
                try!(self.assemble_candidates_from_impls(obligation, &mut candidates));

                // For other types, we'll use the builtin rules.
                try!(self.assemble_builtin_bound_candidates(ty::BoundCopy,
                                                            obligation,
                                                            &mut candidates));
            }
            Some(bound @ ty::BoundSized) => {
                // Sized is never implementable by end-users, it is
                // always automatically computed.
                try!(self.assemble_builtin_bound_candidates(bound,
                                                            obligation,
                                                            &mut candidates));
            }

            None if self.tcx().lang_items.unsize_trait() ==
                    Some(obligation.predicate.def_id()) => {
                self.assemble_candidates_for_unsizing(obligation, &mut candidates);
            }

            Some(ty::BoundSend) |
            Some(ty::BoundSync) |
            None => {
                try!(self.assemble_closure_candidates(obligation, &mut candidates));
                try!(self.assemble_fn_pointer_candidates(obligation, &mut candidates));
                try!(self.assemble_candidates_from_impls(obligation, &mut candidates));
                self.assemble_candidates_from_object_ty(obligation, &mut candidates);
            }
        }

        self.assemble_candidates_from_projected_tys(obligation, &mut candidates);
        try!(self.assemble_candidates_from_caller_bounds(stack, &mut candidates));
        // Default implementations have lower priority, so we only
        // consider triggering a default if there is no other impl that can apply.
        if candidates.vec.is_empty() {
            try!(self.assemble_candidates_from_default_impls(obligation, &mut candidates));
        }
        debug!("candidate list size: {}", candidates.vec.len());
        Ok(candidates)
    }

    fn assemble_candidates_from_projected_tys(&mut self,
                                              obligation: &TraitObligation<'tcx>,
                                              candidates: &mut SelectionCandidateSet<'tcx>)
    {
        debug!("assemble_candidates_for_projected_tys({:?})", obligation);

        // FIXME(#20297) -- just examining the self-type is very simplistic

        // before we go into the whole skolemization thing, just
        // quickly check if the self-type is a projection at all.
        let trait_def_id = match obligation.predicate.0.trait_ref.self_ty().sty {
            ty::TyProjection(ref data) => data.trait_ref.def_id,
            ty::TyInfer(ty::TyVar(_)) => {
                self.tcx().sess.span_bug(obligation.cause.span,
                    "Self=_ should have been handled by assemble_candidates");
            }
            _ => { return; }
        };

        debug!("assemble_candidates_for_projected_tys: trait_def_id={:?}",
               trait_def_id);

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
                skol_trait_predicate={:?} skol_map={:?}",
               skol_trait_predicate,
               skol_map);

        let projection_trait_ref = match skol_trait_predicate.trait_ref.self_ty().sty {
            ty::TyProjection(ref data) => &data.trait_ref,
            _ => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    &format!("match_projection_obligation_against_bounds_from_trait() called \
                              but self-ty not a projection: {:?}",
                             skol_trait_predicate.trait_ref.self_ty()));
            }
        };
        debug!("match_projection_obligation_against_bounds_from_trait: \
                projection_trait_ref={:?}",
               projection_trait_ref);

        let trait_predicates = self.tcx().lookup_predicates(projection_trait_ref.def_id);
        let bounds = trait_predicates.instantiate(self.tcx(), projection_trait_ref.substs);
        debug!("match_projection_obligation_against_bounds_from_trait: \
                bounds={:?}",
               bounds);

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
                matching_bound={:?}",
               matching_bound);
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
                        skol_trait_ref: ty::TraitRef<'tcx>,
                        skol_map: &infer::SkolemizationMap,
                        snapshot: &infer::CombinedSnapshot)
                        -> bool
    {
        assert!(!skol_trait_ref.has_escaping_regions());
        let origin = TypeOrigin::RelateOutputImplTypes(obligation.cause.span);
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
        debug!("assemble_candidates_from_caller_bounds({:?})",
               stack.obligation);

        let all_bounds =
            self.param_env().caller_bounds
                            .iter()
                            .filter_map(|o| o.to_opt_poly_trait_ref());

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
                                 -> EvaluationResult
    {
        self.infcx().probe(move |_| {
            match self.match_where_clause_trait_ref(stack.obligation, where_clause_trait_ref) {
                Ok(obligations) => {
                    self.evaluate_predicates_recursively(stack.list(), obligations.iter())
                }
                Err(()) => EvaluatedToErr
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

        // ok to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters
        let self_ty = *obligation.self_ty().skip_binder();
        let (closure_def_id, substs) = match self_ty.sty {
            ty::TyClosure(id, ref substs) => (id, substs),
            ty::TyInfer(ty::TyVar(_)) => {
                debug!("assemble_unboxed_closure_candidates: ambiguous self-type");
                candidates.ambiguous = true;
                return Ok(());
            }
            _ => { return Ok(()); }
        };

        debug!("assemble_unboxed_candidates: self_ty={:?} kind={:?} obligation={:?}",
               self_ty,
               kind,
               obligation);

        match self.infcx.closure_kind(closure_def_id) {
            Some(closure_kind) => {
                debug!("assemble_unboxed_candidates: closure_kind = {:?}", closure_kind);
                if closure_kind.extends(kind) {
                    candidates.vec.push(ClosureCandidate(closure_def_id, substs));
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
        // We provide impl of all fn traits for fn pointers.
        if self.tcx().lang_items.fn_trait_kind(obligation.predicate.def_id()).is_none() {
            return Ok(());
        }

        // ok to skip binder because what we are inspecting doesn't involve bound regions
        let self_ty = *obligation.self_ty().skip_binder();
        match self_ty.sty {
            ty::TyInfer(ty::TyVar(_)) => {
                debug!("assemble_fn_pointer_candidates: ambiguous self-type");
                candidates.ambiguous = true; // could wind up being a fn() type
            }

            // provide an impl, but only for suitable `fn` pointers
            ty::TyBareFn(_, &ty::BareFnTy {
                unsafety: hir::Unsafety::Normal,
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
        debug!("assemble_candidates_from_impls(obligation={:?})", obligation);

        let def = self.tcx().lookup_trait_def(obligation.predicate.def_id());

        def.for_each_relevant_impl(
            self.tcx(),
            obligation.predicate.0.trait_ref.self_ty(),
            |impl_def_id| {
                self.infcx.probe(|snapshot| {
                    if let Ok(_) = self.match_impl(impl_def_id, obligation, snapshot) {
                        candidates.vec.push(ImplCandidate(impl_def_id));
                    }
                });
            }
        );

        Ok(())
    }

    fn assemble_candidates_from_default_impls(&mut self,
                                              obligation: &TraitObligation<'tcx>,
                                              candidates: &mut SelectionCandidateSet<'tcx>)
                                              -> Result<(), SelectionError<'tcx>>
    {
        // OK to skip binder here because the tests we do below do not involve bound regions
        let self_ty = *obligation.self_ty().skip_binder();
        debug!("assemble_candidates_from_default_impls(self_ty={:?})", self_ty);

        let def_id = obligation.predicate.def_id();

        if self.tcx().trait_has_default_impl(def_id) {
            match self_ty.sty {
                ty::TyTrait(..) => {
                    // For object types, we don't know what the closed
                    // over types are. For most traits, this means we
                    // conservatively say nothing; a candidate may be
                    // added by `assemble_candidates_from_object_ty`.
                    // However, for the kind of magic reflect trait,
                    // we consider it to be implemented even for
                    // object types, because it just lets you reflect
                    // onto the object type, not into the object's
                    // interior.
                    if self.tcx().has_attr(def_id, "rustc_reflect_like") {
                        candidates.vec.push(DefaultImplObjectCandidate(def_id));
                    }
                }
                ty::TyParam(..) |
                ty::TyProjection(..) => {
                    // In these cases, we don't know what the actual
                    // type is.  Therefore, we cannot break it down
                    // into its constituent types. So we don't
                    // consider the `..` impl but instead just add no
                    // candidates: this means that typeck will only
                    // succeed if there is another reason to believe
                    // that this obligation holds. That could be a
                    // where-clause or, in the case of an object type,
                    // it could be that the object type lists the
                    // trait (e.g. `Foo+Send : Send`). See
                    // `compile-fail/typeck-default-trait-impl-send-param.rs`
                    // for an example of a test case that exercises
                    // this path.
                }
                ty::TyInfer(ty::TyVar(_)) => {
                    // the defaulted impl might apply, we don't know
                    candidates.ambiguous = true;
                }
                _ => {
                    candidates.vec.push(DefaultImplCandidate(def_id.clone()))
                }
            }
        }

        Ok(())
    }

    /// Search for impls that might apply to `obligation`.
    fn assemble_candidates_from_object_ty(&mut self,
                                          obligation: &TraitObligation<'tcx>,
                                          candidates: &mut SelectionCandidateSet<'tcx>)
    {
        debug!("assemble_candidates_from_object_ty(self_ty={:?})",
               obligation.self_ty().skip_binder());

        // Object-safety candidates are only applicable to object-safe
        // traits. Including this check is useful because it helps
        // inference in cases of traits like `BorrowFrom`, which are
        // not object-safe, and which rely on being able to infer the
        // self-type from one of the other inputs. Without this check,
        // these cases wind up being considered ambiguous due to a
        // (spurious) ambiguity introduced here.
        let predicate_trait_ref = obligation.predicate.to_poly_trait_ref();
        if !object_safety::is_object_safe(self.tcx(), predicate_trait_ref.def_id()) {
            return;
        }

        self.infcx.commit_if_ok(|snapshot| {
            let (self_ty, _) =
                self.infcx().skolemize_late_bound_regions(&obligation.self_ty(), snapshot);
            let poly_trait_ref = match self_ty.sty {
                ty::TyTrait(ref data) => {
                    match self.tcx().lang_items.to_builtin_kind(obligation.predicate.def_id()) {
                        Some(bound @ ty::BoundSend) | Some(bound @ ty::BoundSync) => {
                            if data.bounds.builtin_bounds.contains(&bound) {
                                debug!("assemble_candidates_from_object_ty: matched builtin bound, \
                                        pushing candidate");
                                candidates.vec.push(BuiltinObjectCandidate);
                                return Ok(());
                            }
                        }
                        _ => {}
                    }

                    data.principal_trait_ref_with_self_ty(self.tcx(), self_ty)
                }
                ty::TyInfer(ty::TyVar(_)) => {
                    debug!("assemble_candidates_from_object_ty: ambiguous");
                    candidates.ambiguous = true; // could wind up being an object type
                    return Ok(());
                }
                _ => {
                    return Ok(());
                }
            };

            debug!("assemble_candidates_from_object_ty: poly_trait_ref={:?}",
                   poly_trait_ref);

            // Count only those upcast versions that match the trait-ref
            // we are looking for. Specifically, do not only check for the
            // correct trait, but also the correct type parameters.
            // For example, we may be trying to upcast `Foo` to `Bar<i32>`,
            // but `Foo` is declared as `trait Foo : Bar<u32>`.
            let upcast_trait_refs =
                util::supertraits(self.tcx(), poly_trait_ref)
                .filter(|upcast_trait_ref| {
                    self.infcx.probe(|_| {
                        let upcast_trait_ref = upcast_trait_ref.clone();
                        self.match_poly_trait_ref(obligation, upcast_trait_ref).is_ok()
                    })
                })
                .count();

            if upcast_trait_refs > 1 {
                // can be upcast in many ways; need more type information
                candidates.ambiguous = true;
            } else if upcast_trait_refs == 1 {
                candidates.vec.push(ObjectCandidate);
            }

            Ok::<(),()>(())
        }).unwrap();
    }

    /// Search for unsizing that might apply to `obligation`.
    fn assemble_candidates_for_unsizing(&mut self,
                                        obligation: &TraitObligation<'tcx>,
                                        candidates: &mut SelectionCandidateSet<'tcx>) {
        // We currently never consider higher-ranked obligations e.g.
        // `for<'a> &'a T: Unsize<Trait+'a>` to be implemented. This is not
        // because they are a priori invalid, and we could potentially add support
        // for them later, it's just that there isn't really a strong need for it.
        // A `T: Unsize<U>` obligation is always used as part of a `T: CoerceUnsize<U>`
        // impl, and those are generally applied to concrete types.
        //
        // That said, one might try to write a fn with a where clause like
        //     for<'a> Foo<'a, T>: Unsize<Foo<'a, Trait>>
        // where the `'a` is kind of orthogonal to the relevant part of the `Unsize`.
        // Still, you'd be more likely to write that where clause as
        //     T: Trait
        // so it seems ok if we (conservatively) fail to accept that `Unsize`
        // obligation above. Should be possible to extend this in the future.
        let source = match self.tcx().no_late_bound_regions(&obligation.self_ty()) {
            Some(t) => t,
            None => {
                // Don't add any candidates if there are bound regions.
                return;
            }
        };
        let target = obligation.predicate.0.input_types()[0];

        debug!("assemble_candidates_for_unsizing(source={:?}, target={:?})",
               source, target);

        let may_apply = match (&source.sty, &target.sty) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::TyTrait(ref data_a), &ty::TyTrait(ref data_b)) => {
                // Upcasts permit two things:
                //
                // 1. Dropping builtin bounds, e.g. `Foo+Send` to `Foo`
                // 2. Tightening the region bound, e.g. `Foo+'a` to `Foo+'b` if `'a : 'b`
                //
                // Note that neither of these changes requires any
                // change at runtime.  Eventually this will be
                // generalized.
                //
                // We always upcast when we can because of reason
                // #2 (region bounds).
                data_a.principal.def_id() == data_a.principal.def_id() &&
                data_a.bounds.builtin_bounds.is_superset(&data_b.bounds.builtin_bounds)
            }

            // T -> Trait.
            (_, &ty::TyTrait(_)) => true,

            // Ambiguous handling is below T -> Trait, because inference
            // variables can still implement Unsize<Trait> and nested
            // obligations will have the final say (likely deferred).
            (&ty::TyInfer(ty::TyVar(_)), _) |
            (_, &ty::TyInfer(ty::TyVar(_))) => {
                debug!("assemble_candidates_for_unsizing: ambiguous");
                candidates.ambiguous = true;
                false
            }

            // [T; n] -> [T].
            (&ty::TyArray(_, _), &ty::TySlice(_)) => true,

            // Struct<T> -> Struct<U>.
            (&ty::TyStruct(def_id_a, _), &ty::TyStruct(def_id_b, _)) => {
                def_id_a == def_id_b
            }

            _ => false
        };

        if may_apply {
            candidates.vec.push(BuiltinUnsizeCandidate);
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

    /// Returns true if `candidate_i` should be dropped in favor of
    /// `candidate_j`.  Generally speaking we will drop duplicate
    /// candidates and prefer where-clause candidates.
    /// Returns true if `victim` should be dropped in favor of
    /// `other`.  Generally speaking we will drop duplicate
    /// candidates and prefer where-clause candidates.
    ///
    /// See the comment for "SelectionCandidate" for more details.
    fn candidate_should_be_dropped_in_favor_of<'o>(&mut self,
                                                   victim: &SelectionCandidate<'tcx>,
                                                   other: &SelectionCandidate<'tcx>)
                                                   -> bool
    {
        if victim == other {
            return true;
        }

        match other {
            &ObjectCandidate |
            &ParamCandidate(_) | &ProjectionCandidate => match victim {
                &DefaultImplCandidate(..) => {
                    self.tcx().sess.bug(
                        "default implementations shouldn't be recorded \
                         when there are other valid candidates");
                }
                &ImplCandidate(..) |
                &ClosureCandidate(..) |
                &FnPointerCandidate |
                &BuiltinObjectCandidate |
                &BuiltinUnsizeCandidate |
                &DefaultImplObjectCandidate(..) |
                &BuiltinCandidate(..) => {
                    // We have a where-clause so don't go around looking
                    // for impls.
                    true
                }
                &ObjectCandidate |
                &ProjectionCandidate => {
                    // Arbitrarily give param candidates priority
                    // over projection and object candidates.
                    true
                },
                &ParamCandidate(..) => false,
            },
            _ => false
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
                                             obligation: &TraitObligation<'tcx>,
                                             candidates: &mut SelectionCandidateSet<'tcx>)
                                             -> Result<(),SelectionError<'tcx>>
    {
        match self.builtin_bound(bound, obligation) {
            Ok(If(..)) => {
                debug!("builtin_bound: bound={:?}",
                       bound);
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
            ty::TyInfer(ty::IntVar(_)) |
            ty::TyInfer(ty::FloatVar(_)) |
            ty::TyUint(_) |
            ty::TyInt(_) |
            ty::TyBool |
            ty::TyFloat(_) |
            ty::TyBareFn(..) |
            ty::TyChar => {
                // safe for everything
                ok_if(Vec::new())
            }

            ty::TyBox(_) => {  // Box<T>
                match bound {
                    ty::BoundCopy => Err(Unimplemented),

                    ty::BoundSized => ok_if(Vec::new()),

                    ty::BoundSync | ty::BoundSend => {
                        self.tcx().sess.bug("Send/Sync shouldn't occur in builtin_bounds()");
                    }
                }
            }

            ty::TyRawPtr(..) => {     // *const T, *mut T
                match bound {
                    ty::BoundCopy | ty::BoundSized => ok_if(Vec::new()),

                    ty::BoundSync | ty::BoundSend => {
                        self.tcx().sess.bug("Send/Sync shouldn't occur in builtin_bounds()");
                    }
                }
            }

            ty::TyTrait(ref data) => {
                match bound {
                    ty::BoundSized => Err(Unimplemented),
                    ty::BoundCopy => {
                        if data.bounds.builtin_bounds.contains(&bound) {
                            ok_if(Vec::new())
                        } else {
                            // Recursively check all supertraits to find out if any further
                            // bounds are required and thus we must fulfill.
                            let principal =
                                data.principal_trait_ref_with_self_ty(self.tcx(),
                                                                      self.tcx().types.err);
                            let copy_def_id = obligation.predicate.def_id();
                            for tr in util::supertraits(self.tcx(), principal) {
                                if tr.def_id() == copy_def_id {
                                    return ok_if(Vec::new())
                                }
                            }

                            Err(Unimplemented)
                        }
                    }
                    ty::BoundSync | ty::BoundSend => {
                        self.tcx().sess.bug("Send/Sync shouldn't occur in builtin_bounds()");
                    }
                }
            }

            ty::TyRef(_, ty::TypeAndMut { ty: _, mutbl }) => {
                // &mut T or &T
                match bound {
                    ty::BoundCopy => {
                        match mutbl {
                            // &mut T is affine and hence never `Copy`
                            hir::MutMutable => Err(Unimplemented),

                            // &T is always copyable
                            hir::MutImmutable => ok_if(Vec::new()),
                        }
                    }

                    ty::BoundSized => ok_if(Vec::new()),

                    ty::BoundSync | ty::BoundSend => {
                        self.tcx().sess.bug("Send/Sync shouldn't occur in builtin_bounds()");
                    }
                }
            }

            ty::TyArray(element_ty, _) => {
                // [T; n]
                match bound {
                    ty::BoundCopy => ok_if(vec![element_ty]),
                    ty::BoundSized => ok_if(Vec::new()),
                    ty::BoundSync | ty::BoundSend => {
                        self.tcx().sess.bug("Send/Sync shouldn't occur in builtin_bounds()");
                    }
                }
            }

            ty::TyStr | ty::TySlice(_) => {
                match bound {
                    ty::BoundSync | ty::BoundSend => {
                        self.tcx().sess.bug("Send/Sync shouldn't occur in builtin_bounds()");
                    }

                    ty::BoundCopy | ty::BoundSized => Err(Unimplemented),
                }
            }

            // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
            ty::TyTuple(ref tys) => ok_if(tys.clone()),

            ty::TyClosure(_, ref substs) => {
                // FIXME -- This case is tricky. In the case of by-ref
                // closures particularly, we need the results of
                // inference to decide how to reflect the type of each
                // upvar (the upvar may have type `T`, but the runtime
                // type could be `&mut`, `&`, or just `T`). For now,
                // though, we'll do this unsoundly and assume that all
                // captures are by value. Really what we ought to do
                // is reserve judgement and then intertwine this
                // analysis with closure inference.

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
                    return ok_if(Vec::new());
                }

                ok_if(substs.upvar_tys.clone())
            }

            ty::TyStruct(def, substs) | ty::TyEnum(def, substs) => {
                let types: Vec<Ty> = def.all_fields().map(|f| {
                    f.ty(self.tcx(), substs)
                }).collect();
                nominal(bound, types)
            }

            ty::TyProjection(_) | ty::TyParam(_) => {
                // Note: A type parameter is only considered to meet a
                // particular bound if there is a where clause telling
                // us that it does, and that case is handled by
                // `assemble_candidates_from_caller_bounds()`.
                Ok(ParameterBuiltin)
            }

            ty::TyInfer(ty::TyVar(_)) => {
                // Unbound type variable. Might or might not have
                // applicable impls and so forth, depending on what
                // those type variables wind up being bound to.
                debug!("assemble_builtin_bound_candidates: ambiguous builtin");
                Ok(AmbiguousBuiltin)
            }

            ty::TyError => ok_if(Vec::new()),

            ty::TyInfer(ty::FreshTy(_))
            | ty::TyInfer(ty::FreshIntTy(_))
            | ty::TyInfer(ty::FreshFloatTy(_)) => {
                self.tcx().sess.bug(
                    &format!(
                        "asked to assemble builtin bounds of unexpected type: {:?}",
                        self_ty));
            }
        };

        fn ok_if<'tcx>(v: Vec<Ty<'tcx>>)
                       -> Result<BuiltinBoundConditions<'tcx>, SelectionError<'tcx>> {
            Ok(If(ty::Binder(v)))
        }

        fn nominal<'cx, 'tcx>(bound: ty::BuiltinBound,
                              types: Vec<Ty<'tcx>>)
                              -> Result<BuiltinBoundConditions<'tcx>, SelectionError<'tcx>>
        {
            // First check for markers and other nonsense.
            match bound {
                // Fallback to whatever user-defined impls exist in this case.
                ty::BoundCopy => Ok(ParameterBuiltin),

                // Sized if all the component types are sized.
                ty::BoundSized => ok_if(types),

                // Shouldn't be coming through here.
                ty::BoundSend | ty::BoundSync => unreachable!(),
            }
        }
    }

    /// For default impls, we need to break apart a type into its
    /// "constituent types" -- meaning, the types that it contains.
    ///
    /// Here are some (simple) examples:
    ///
    /// ```
    /// (i32, u32) -> [i32, u32]
    /// Foo where struct Foo { x: i32, y: u32 } -> [i32, u32]
    /// Bar<i32> where struct Bar<T> { x: T, y: u32 } -> [i32, u32]
    /// Zed<i32> where enum Zed { A(T), B(u32) } -> [i32, u32]
    /// ```
    fn constituent_types_for_ty(&self, t: Ty<'tcx>) -> Vec<Ty<'tcx>> {
        match t.sty {
            ty::TyUint(_) |
            ty::TyInt(_) |
            ty::TyBool |
            ty::TyFloat(_) |
            ty::TyBareFn(..) |
            ty::TyStr |
            ty::TyError |
            ty::TyInfer(ty::IntVar(_)) |
            ty::TyInfer(ty::FloatVar(_)) |
            ty::TyChar => {
                Vec::new()
            }

            ty::TyTrait(..) |
            ty::TyParam(..) |
            ty::TyProjection(..) |
            ty::TyInfer(ty::TyVar(_)) |
            ty::TyInfer(ty::FreshTy(_)) |
            ty::TyInfer(ty::FreshIntTy(_)) |
            ty::TyInfer(ty::FreshFloatTy(_)) => {
                self.tcx().sess.bug(
                    &format!(
                        "asked to assemble constituent types of unexpected type: {:?}",
                        t));
            }

            ty::TyBox(referent_ty) => {  // Box<T>
                vec![referent_ty]
            }

            ty::TyRawPtr(ty::TypeAndMut { ty: element_ty, ..}) |
            ty::TyRef(_, ty::TypeAndMut { ty: element_ty, ..}) => {
                vec![element_ty]
            },

            ty::TyArray(element_ty, _) | ty::TySlice(element_ty) => {
                vec![element_ty]
            }

            ty::TyTuple(ref tys) => {
                // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
                tys.clone()
            }

            ty::TyClosure(_, ref substs) => {
                // FIXME(#27086). We are invariant w/r/t our
                // substs.func_substs, but we don't see them as
                // constituent types; this seems RIGHT but also like
                // something that a normal type couldn't simulate. Is
                // this just a gap with the way that PhantomData and
                // OIBIT interact? That is, there is no way to say
                // "make me invariant with respect to this TYPE, but
                // do not act as though I can reach it"
                substs.upvar_tys.clone()
            }

            // for `PhantomData<T>`, we pass `T`
            ty::TyStruct(def, substs) if def.is_phantom_data() => {
                substs.types.get_slice(TypeSpace).to_vec()
            }

            ty::TyStruct(def, substs) | ty::TyEnum(def, substs) => {
                def.all_fields()
                    .map(|f| f.ty(self.tcx(), substs))
                    .collect()
            }
        }
    }

    fn collect_predicates_for_types(&mut self,
                                    obligation: &TraitObligation<'tcx>,
                                    trait_def_id: DefId,
                                    types: ty::Binder<Vec<Ty<'tcx>>>)
                                    -> Vec<PredicateObligation<'tcx>>
    {
        let derived_cause = match self.tcx().lang_items.to_builtin_kind(trait_def_id) {
            Some(_) => {
                self.derived_cause(obligation, BuiltinDerivedObligation)
            },
            None => {
                self.derived_cause(obligation, ImplDerivedObligation)
            }
        };

        // Because the types were potentially derived from
        // higher-ranked obligations they may reference late-bound
        // regions. For example, `for<'a> Foo<&'a int> : Copy` would
        // yield a type like `for<'a> &'a int`. In general, we
        // maintain the invariant that we never manipulate bound
        // regions, so we have to process these bound regions somehow.
        //
        // The strategy is to:
        //
        // 1. Instantiate those regions to skolemized regions (e.g.,
        //    `for<'a> &'a int` becomes `&0 int`.
        // 2. Produce something like `&'0 int : Copy`
        // 3. Re-bind the regions back to `for<'a> &'a int : Copy`

        // Move the binder into the individual types
        let bound_types: Vec<ty::Binder<Ty<'tcx>>> =
            types.skip_binder()
                 .iter()
                 .map(|&nested_ty| ty::Binder(nested_ty))
                 .collect();

        // For each type, produce a vector of resulting obligations
        let obligations: Result<Vec<Vec<_>>, _> = bound_types.iter().map(|nested_ty| {
            self.infcx.commit_if_ok(|snapshot| {
                let (skol_ty, skol_map) =
                    self.infcx().skolemize_late_bound_regions(nested_ty, snapshot);
                let Normalized { value: normalized_ty, mut obligations } =
                    project::normalize_with_depth(self,
                                                  obligation.cause.clone(),
                                                  obligation.recursion_depth + 1,
                                                  &skol_ty);
                let skol_obligation =
                    util::predicate_for_trait_def(self.tcx(),
                                                  derived_cause.clone(),
                                                  trait_def_id,
                                                  obligation.recursion_depth + 1,
                                                  normalized_ty,
                                                  vec![]);
                obligations.push(skol_obligation);
                Ok(self.infcx().plug_leaks(skol_map, snapshot, &obligations))
            })
        }).collect();

        // Flatten those vectors (couldn't do it above due `collect`)
        match obligations {
            Ok(obligations) => obligations.into_iter().flat_map(|o| o).collect(),
            Err(ErrorReported) => Vec::new(),
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // CONFIRMATION
    //
    // Confirmation unifies the output type parameters of the trait
    // with the values found in the obligation, possibly yielding a
    // type error.  See `README.md` for more details.

    fn confirm_candidate(&mut self,
                         obligation: &TraitObligation<'tcx>,
                         candidate: SelectionCandidate<'tcx>)
                         -> Result<Selection<'tcx>,SelectionError<'tcx>>
    {
        debug!("confirm_candidate({:?}, {:?})",
               obligation,
               candidate);

        match candidate {
            BuiltinCandidate(builtin_bound) => {
                Ok(VtableBuiltin(
                    try!(self.confirm_builtin_candidate(obligation, builtin_bound))))
            }

            ParamCandidate(param) => {
                let obligations = self.confirm_param_candidate(obligation, param);
                Ok(VtableParam(obligations))
            }

            DefaultImplCandidate(trait_def_id) => {
                let data = self.confirm_default_impl_candidate(obligation, trait_def_id);
                Ok(VtableDefaultImpl(data))
            }

            DefaultImplObjectCandidate(trait_def_id) => {
                let data = self.confirm_default_impl_object_candidate(obligation, trait_def_id);
                Ok(VtableDefaultImpl(data))
            }

            ImplCandidate(impl_def_id) => {
                let vtable_impl =
                    try!(self.confirm_impl_candidate(obligation, impl_def_id));
                Ok(VtableImpl(vtable_impl))
            }

            ClosureCandidate(closure_def_id, substs) => {
                let vtable_closure =
                    try!(self.confirm_closure_candidate(obligation, closure_def_id, substs));
                Ok(VtableClosure(vtable_closure))
            }

            BuiltinObjectCandidate => {
                // This indicates something like `(Trait+Send) :
                // Send`. In this case, we know that this holds
                // because that's what the object type is telling us,
                // and there's really no additional obligations to
                // prove and no types in particular to unify etc.
                Ok(VtableParam(Vec::new()))
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

            BuiltinUnsizeCandidate => {
                let data = try!(self.confirm_builtin_unsize_candidate(obligation));
                Ok(VtableBuiltin(data))
            }
        }
    }

    fn confirm_projection_candidate(&mut self,
                                    obligation: &TraitObligation<'tcx>)
    {
        let _: Result<(),()> =
            self.infcx.commit_if_ok(|snapshot| {
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
        debug!("confirm_param_candidate({:?},{:?})",
               obligation,
               param);

        // During evaluation, we already checked that this
        // where-clause trait-ref could be unified with the obligation
        // trait-ref. Repeat that unification now without any
        // transactional boundary; it should not fail.
        match self.match_where_clause_trait_ref(obligation, param.clone()) {
            Ok(obligations) => obligations,
            Err(()) => {
                self.tcx().sess.bug(
                    &format!("Where clause `{:?}` was applicable to `{:?}` but now is not",
                             param,
                             obligation));
            }
        }
    }

    fn confirm_builtin_candidate(&mut self,
                                 obligation: &TraitObligation<'tcx>,
                                 bound: ty::BuiltinBound)
                                 -> Result<VtableBuiltinData<PredicateObligation<'tcx>>,
                                           SelectionError<'tcx>>
    {
        debug!("confirm_builtin_candidate({:?})",
               obligation);

        match try!(self.builtin_bound(bound, obligation)) {
            If(nested) => Ok(self.vtable_builtin_data(obligation, bound, nested)),
            AmbiguousBuiltin | ParameterBuiltin => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    &format!("builtin bound for {:?} was ambig",
                            obligation));
            }
        }
    }

    fn vtable_builtin_data(&mut self,
                           obligation: &TraitObligation<'tcx>,
                           bound: ty::BuiltinBound,
                           nested: ty::Binder<Vec<Ty<'tcx>>>)
                           -> VtableBuiltinData<PredicateObligation<'tcx>>
    {
        let trait_def = match self.tcx().lang_items.from_builtin_kind(bound) {
            Ok(def_id) => def_id,
            Err(_) => {
                self.tcx().sess.bug("builtin trait definition not found");
            }
        };

        let obligations = self.collect_predicates_for_types(obligation, trait_def, nested);

        debug!("vtable_builtin_data: obligations={:?}",
               obligations);

        VtableBuiltinData { nested: obligations }
    }

    /// This handles the case where a `impl Foo for ..` impl is being used.
    /// The idea is that the impl applies to `X : Foo` if the following conditions are met:
    ///
    /// 1. For each constituent type `Y` in `X`, `Y : Foo` holds
    /// 2. For each where-clause `C` declared on `Foo`, `[Self => X] C` holds.
    fn confirm_default_impl_candidate(&mut self,
                                      obligation: &TraitObligation<'tcx>,
                                      trait_def_id: DefId)
                                      -> VtableDefaultImplData<PredicateObligation<'tcx>>
    {
        debug!("confirm_default_impl_candidate({:?}, {:?})",
               obligation,
               trait_def_id);

        // binder is moved below
        let self_ty = self.infcx.shallow_resolve(obligation.predicate.skip_binder().self_ty());
        let types = self.constituent_types_for_ty(self_ty);
        self.vtable_default_impl(obligation, trait_def_id, ty::Binder(types))
    }

    fn confirm_default_impl_object_candidate(&mut self,
                                             obligation: &TraitObligation<'tcx>,
                                             trait_def_id: DefId)
                                             -> VtableDefaultImplData<PredicateObligation<'tcx>>
    {
        debug!("confirm_default_impl_object_candidate({:?}, {:?})",
               obligation,
               trait_def_id);

        assert!(self.tcx().has_attr(trait_def_id, "rustc_reflect_like"));

        // OK to skip binder, it is reintroduced below
        let self_ty = self.infcx.shallow_resolve(obligation.predicate.skip_binder().self_ty());
        match self_ty.sty {
            ty::TyTrait(ref data) => {
                // OK to skip the binder, it is reintroduced below
                let input_types = data.principal.skip_binder().substs.types.get_slice(TypeSpace);
                let assoc_types = data.bounds.projection_bounds
                                             .iter()
                                             .map(|pb| pb.skip_binder().ty);
                let all_types: Vec<_> = input_types.iter().cloned()
                                                          .chain(assoc_types)
                                                          .collect();

                // reintroduce the two binding levels we skipped, then flatten into one
                let all_types = ty::Binder(ty::Binder(all_types));
                let all_types = self.tcx().flatten_late_bound_regions(&all_types);

                self.vtable_default_impl(obligation, trait_def_id, all_types)
            }
            _ => {
                self.tcx().sess.bug(
                    &format!(
                        "asked to confirm default object implementation for non-object type: {:?}",
                        self_ty));
            }
        }
    }

    /// See `confirm_default_impl_candidate`
    fn vtable_default_impl(&mut self,
                           obligation: &TraitObligation<'tcx>,
                           trait_def_id: DefId,
                           nested: ty::Binder<Vec<Ty<'tcx>>>)
                           -> VtableDefaultImplData<PredicateObligation<'tcx>>
    {
        debug!("vtable_default_impl_data: nested={:?}", nested);

        let mut obligations = self.collect_predicates_for_types(obligation,
                                                                trait_def_id,
                                                                nested);

        let trait_obligations: Result<Vec<_>,()> = self.infcx.commit_if_ok(|snapshot| {
            let poly_trait_ref = obligation.predicate.to_poly_trait_ref();
            let (trait_ref, skol_map) =
                self.infcx().skolemize_late_bound_regions(&poly_trait_ref, snapshot);
            Ok(self.impl_or_trait_obligations(obligation.cause.clone(),
                                              obligation.recursion_depth + 1,
                                              trait_def_id,
                                              &trait_ref.substs,
                                              skol_map,
                                              snapshot))
        });

        // no Errors in that code above
        obligations.append(&mut trait_obligations.unwrap());

        debug!("vtable_default_impl_data: obligations={:?}", obligations);

        VtableDefaultImplData {
            trait_def_id: trait_def_id,
            nested: obligations
        }
    }

    fn confirm_impl_candidate(&mut self,
                              obligation: &TraitObligation<'tcx>,
                              impl_def_id: DefId)
                              -> Result<VtableImplData<'tcx, PredicateObligation<'tcx>>,
                                        SelectionError<'tcx>>
    {
        debug!("confirm_impl_candidate({:?},{:?})",
               obligation,
               impl_def_id);

        // First, create the substitutions by matching the impl again,
        // this time not in a probe.
        self.infcx.commit_if_ok(|snapshot| {
            let (substs, skol_map) =
                self.rematch_impl(impl_def_id, obligation,
                                  snapshot);
            debug!("confirm_impl_candidate substs={:?}", substs);
            Ok(self.vtable_impl(impl_def_id, substs, obligation.cause.clone(),
                                obligation.recursion_depth + 1, skol_map, snapshot))
        })
    }

    fn vtable_impl(&mut self,
                   impl_def_id: DefId,
                   mut substs: Normalized<'tcx, Substs<'tcx>>,
                   cause: ObligationCause<'tcx>,
                   recursion_depth: usize,
                   skol_map: infer::SkolemizationMap,
                   snapshot: &infer::CombinedSnapshot)
                   -> VtableImplData<'tcx, PredicateObligation<'tcx>>
    {
        debug!("vtable_impl(impl_def_id={:?}, substs={:?}, recursion_depth={}, skol_map={:?})",
               impl_def_id,
               substs,
               recursion_depth,
               skol_map);

        let mut impl_obligations =
            self.impl_or_trait_obligations(cause,
                                           recursion_depth,
                                           impl_def_id,
                                           &substs.value,
                                           skol_map,
                                           snapshot);

        debug!("vtable_impl: impl_def_id={:?} impl_obligations={:?}",
               impl_def_id,
               impl_obligations);

        // Because of RFC447, the impl-trait-ref and obligations
        // are sufficient to determine the impl substs, without
        // relying on projections in the impl-trait-ref.
        //
        // e.g. `impl<U: Tr, V: Iterator<Item=U>> Foo<<U as Tr>::T> for V`
        impl_obligations.append(&mut substs.obligations);

        VtableImplData { impl_def_id: impl_def_id,
                         substs: substs.value,
                         nested: impl_obligations }
    }

    fn confirm_object_candidate(&mut self,
                                obligation: &TraitObligation<'tcx>)
                                -> VtableObjectData<'tcx>
    {
        debug!("confirm_object_candidate({:?})",
               obligation);

        // FIXME skipping binder here seems wrong -- we should
        // probably flatten the binder from the obligation and the
        // binder from the object. Have to try to make a broken test
        // case that results. -nmatsakis
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let poly_trait_ref = match self_ty.sty {
            ty::TyTrait(ref data) => {
                data.principal_trait_ref_with_self_ty(self.tcx(), self_ty)
            }
            _ => {
                self.tcx().sess.span_bug(obligation.cause.span,
                                         "object candidate with non-object");
            }
        };

        let mut upcast_trait_ref = None;
        let vtable_base;

        {
            // We want to find the first supertrait in the list of
            // supertraits that we can unify with, and do that
            // unification. We know that there is exactly one in the list
            // where we can unify because otherwise select would have
            // reported an ambiguity. (When we do find a match, also
            // record it for later.)
            let nonmatching =
                util::supertraits(self.tcx(), poly_trait_ref)
                .take_while(|&t| {
                    match
                        self.infcx.commit_if_ok(
                            |_| self.match_poly_trait_ref(obligation, t))
                    {
                        Ok(_) => { upcast_trait_ref = Some(t); false }
                        Err(_) => { true }
                    }
                });

            // Additionally, for each of the nonmatching predicates that
            // we pass over, we sum up the set of number of vtable
            // entries, so that we can compute the offset for the selected
            // trait.
            vtable_base =
                nonmatching.map(|t| util::count_own_vtable_entries(self.tcx(), t))
                           .sum();

        }

        VtableObjectData {
            upcast_trait_ref: upcast_trait_ref.unwrap(),
            vtable_base: vtable_base,
        }
    }

    fn confirm_fn_pointer_candidate(&mut self,
                                    obligation: &TraitObligation<'tcx>)
                                    -> Result<ty::Ty<'tcx>,SelectionError<'tcx>>
    {
        debug!("confirm_fn_pointer_candidate({:?})",
               obligation);

        // ok to skip binder; it is reintroduced below
        let self_ty = self.infcx.shallow_resolve(*obligation.self_ty().skip_binder());
        let sig = self_ty.fn_sig();
        let trait_ref =
            util::closure_trait_ref_and_return_type(self.tcx(),
                                                    obligation.predicate.def_id(),
                                                    self_ty,
                                                    sig,
                                                    util::TupleArgumentsFlag::Yes)
            .map_bound(|(trait_ref, _)| trait_ref);

        try!(self.confirm_poly_trait_refs(obligation.cause.clone(),
                                          obligation.predicate.to_poly_trait_ref(),
                                          trait_ref));
        Ok(self_ty)
    }

    fn confirm_closure_candidate(&mut self,
                                 obligation: &TraitObligation<'tcx>,
                                 closure_def_id: DefId,
                                 substs: &ty::ClosureSubsts<'tcx>)
                                 -> Result<VtableClosureData<'tcx, PredicateObligation<'tcx>>,
                                           SelectionError<'tcx>>
    {
        debug!("confirm_closure_candidate({:?},{:?},{:?})",
               obligation,
               closure_def_id,
               substs);

        let Normalized {
            value: trait_ref,
            obligations
        } = self.closure_trait_ref(obligation, closure_def_id, substs);

        debug!("confirm_closure_candidate(closure_def_id={:?}, trait_ref={:?}, obligations={:?})",
               closure_def_id,
               trait_ref,
               obligations);

        try!(self.confirm_poly_trait_refs(obligation.cause.clone(),
                                          obligation.predicate.to_poly_trait_ref(),
                                          trait_ref));

        Ok(VtableClosureData {
            closure_def_id: closure_def_id,
            substs: substs.clone(),
            nested: obligations
        })
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
    /// Here is an example. Imagine we have a closure expression
    /// and we desugared it so that the type of the expression is
    /// `Closure`, and `Closure` expects an int as argument. Then it
    /// is "as if" the compiler generated this impl:
    ///
    ///     impl Fn(int) for Closure { ... }
    ///
    /// Now imagine our obligation is `Fn(usize) for Closure`. So far
    /// we have matched the self-type `Closure`. At this point we'll
    /// compare the `int` to `usize` and generate an error.
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
        let origin = TypeOrigin::RelateOutputImplTypes(obligation_cause.span);

        let obligation_trait_ref = obligation_trait_ref.clone();
        match self.infcx.sub_poly_trait_refs(false,
                                             origin,
                                             expected_trait_ref.clone(),
                                             obligation_trait_ref.clone()) {
            Ok(()) => Ok(()),
            Err(e) => Err(OutputTypeParameterMismatch(expected_trait_ref, obligation_trait_ref, e))
        }
    }

    fn confirm_builtin_unsize_candidate(&mut self,
                                        obligation: &TraitObligation<'tcx>,)
                                        -> Result<VtableBuiltinData<PredicateObligation<'tcx>>,
                                                  SelectionError<'tcx>> {
        let tcx = self.tcx();

        // assemble_candidates_for_unsizing should ensure there are no late bound
        // regions here. See the comment there for more details.
        let source = self.infcx.shallow_resolve(
            tcx.no_late_bound_regions(&obligation.self_ty()).unwrap());
        let target = self.infcx.shallow_resolve(obligation.predicate.0.input_types()[0]);

        debug!("confirm_builtin_unsize_candidate(source={:?}, target={:?})",
               source, target);

        let mut nested = vec![];
        match (&source.sty, &target.sty) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::TyTrait(ref data_a), &ty::TyTrait(ref data_b)) => {
                // See assemble_candidates_for_unsizing for more info.
                let bounds = ty::ExistentialBounds {
                    region_bound: data_b.bounds.region_bound,
                    builtin_bounds: data_b.bounds.builtin_bounds,
                    projection_bounds: data_a.bounds.projection_bounds.clone(),
                };

                let new_trait = tcx.mk_trait(data_a.principal.clone(), bounds);
                let origin = TypeOrigin::Misc(obligation.cause.span);
                if self.infcx.sub_types(false, origin, new_trait, target).is_err() {
                    return Err(Unimplemented);
                }

                // Register one obligation for 'a: 'b.
                let cause = ObligationCause::new(obligation.cause.span,
                                                 obligation.cause.body_id,
                                                 ObjectCastObligation(target));
                let outlives = ty::OutlivesPredicate(data_a.bounds.region_bound,
                                                     data_b.bounds.region_bound);
                nested.push(Obligation::with_depth(cause,
                                                   obligation.recursion_depth + 1,
                                                   ty::Binder(outlives).to_predicate()));
            }

            // T -> Trait.
            (_, &ty::TyTrait(ref data)) => {
                let object_did = data.principal_def_id();
                if !object_safety::is_object_safe(tcx, object_did) {
                    return Err(TraitNotObjectSafe(object_did));
                }

                let cause = ObligationCause::new(obligation.cause.span,
                                                 obligation.cause.body_id,
                                                 ObjectCastObligation(target));
                let mut push = |predicate| {
                    nested.push(Obligation::with_depth(cause.clone(),
                                                       obligation.recursion_depth + 1,
                                                       predicate));
                };

                // Create the obligation for casting from T to Trait.
                push(data.principal_trait_ref_with_self_ty(tcx, source).to_predicate());

                // We can only make objects from sized types.
                let mut builtin_bounds = data.bounds.builtin_bounds;
                builtin_bounds.insert(ty::BoundSized);

                // Create additional obligations for all the various builtin
                // bounds attached to the object cast. (In other words, if the
                // object type is Foo+Send, this would create an obligation
                // for the Send check.)
                for bound in &builtin_bounds {
                    if let Ok(tr) = util::trait_ref_for_builtin_bound(tcx, bound, source) {
                        push(tr.to_predicate());
                    } else {
                        return Err(Unimplemented);
                    }
                }

                // Create obligations for the projection predicates.
                for bound in data.projection_bounds_with_self_ty(tcx, source) {
                    push(bound.to_predicate());
                }

                // If the type is `Foo+'a`, ensures that the type
                // being cast to `Foo+'a` outlives `'a`:
                let outlives = ty::OutlivesPredicate(source,
                                                     data.bounds.region_bound);
                push(ty::Binder(outlives).to_predicate());
            }

            // [T; n] -> [T].
            (&ty::TyArray(a, _), &ty::TySlice(b)) => {
                let origin = TypeOrigin::Misc(obligation.cause.span);
                if self.infcx.sub_types(false, origin, a, b).is_err() {
                    return Err(Unimplemented);
                }
            }

            // Struct<T> -> Struct<U>.
            (&ty::TyStruct(def, substs_a), &ty::TyStruct(_, substs_b)) => {
                let fields = def
                    .all_fields()
                    .map(|f| f.unsubst_ty())
                    .collect::<Vec<_>>();

                // The last field of the structure has to exist and contain type parameters.
                let field = if let Some(&field) = fields.last() {
                    field
                } else {
                    return Err(Unimplemented);
                };
                let mut ty_params = vec![];
                for ty in field.walk() {
                    if let ty::TyParam(p) = ty.sty {
                        assert!(p.space == TypeSpace);
                        let idx = p.idx as usize;
                        if !ty_params.contains(&idx) {
                            ty_params.push(idx);
                        }
                    }
                }
                if ty_params.is_empty() {
                    return Err(Unimplemented);
                }

                // Replace type parameters used in unsizing with
                // TyError and ensure they do not affect any other fields.
                // This could be checked after type collection for any struct
                // with a potentially unsized trailing field.
                let mut new_substs = substs_a.clone();
                for &i in &ty_params {
                    new_substs.types.get_mut_slice(TypeSpace)[i] = tcx.types.err;
                }
                for &ty in fields.split_last().unwrap().1 {
                    if ty.subst(tcx, &new_substs).references_error() {
                        return Err(Unimplemented);
                    }
                }

                // Extract Field<T> and Field<U> from Struct<T> and Struct<U>.
                let inner_source = field.subst(tcx, substs_a);
                let inner_target = field.subst(tcx, substs_b);

                // Check that the source structure with the target's
                // type parameters is a subtype of the target.
                for &i in &ty_params {
                    let param_b = *substs_b.types.get(TypeSpace, i);
                    new_substs.types.get_mut_slice(TypeSpace)[i] = param_b;
                }
                let new_struct = tcx.mk_struct(def, tcx.mk_substs(new_substs));
                let origin = TypeOrigin::Misc(obligation.cause.span);
                if self.infcx.sub_types(false, origin, new_struct, target).is_err() {
                    return Err(Unimplemented);
                }

                // Construct the nested Field<T>: Unsize<Field<U>> predicate.
                nested.push(util::predicate_for_trait_def(tcx,
                    obligation.cause.clone(),
                    obligation.predicate.def_id(),
                    obligation.recursion_depth + 1,
                    inner_source,
                    vec![inner_target]));
            }

            _ => unreachable!()
        };

        Ok(VtableBuiltinData { nested: nested })
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
                    impl_def_id: DefId,
                    obligation: &TraitObligation<'tcx>,
                    snapshot: &infer::CombinedSnapshot)
                    -> (Normalized<'tcx, Substs<'tcx>>, infer::SkolemizationMap)
    {
        match self.match_impl(impl_def_id, obligation, snapshot) {
            Ok((substs, skol_map)) => (substs, skol_map),
            Err(()) => {
                self.tcx().sess.bug(
                    &format!("Impl {:?} was matchable against {:?} but now is not",
                            impl_def_id,
                            obligation));
            }
        }
    }

    fn match_impl(&mut self,
                  impl_def_id: DefId,
                  obligation: &TraitObligation<'tcx>,
                  snapshot: &infer::CombinedSnapshot)
                  -> Result<(Normalized<'tcx, Substs<'tcx>>,
                             infer::SkolemizationMap), ()>
    {
        let impl_trait_ref = self.tcx().impl_trait_ref(impl_def_id).unwrap();

        // Before we create the substitutions and everything, first
        // consider a "quick reject". This avoids creating more types
        // and so forth that we need to.
        if self.fast_reject_trait_refs(obligation, &impl_trait_ref) {
            return Err(());
        }

        let (skol_obligation, skol_map) = self.infcx().skolemize_late_bound_regions(
            &obligation.predicate,
            snapshot);
        let skol_obligation_trait_ref = skol_obligation.trait_ref;

        let impl_substs = util::fresh_type_vars_for_impl(self.infcx,
                                                         obligation.cause.span,
                                                         impl_def_id);

        let impl_trait_ref = impl_trait_ref.subst(self.tcx(),
                                                  &impl_substs);

        let impl_trait_ref =
            project::normalize_with_depth(self,
                                          obligation.cause.clone(),
                                          obligation.recursion_depth + 1,
                                          &impl_trait_ref);

        debug!("match_impl(impl_def_id={:?}, obligation={:?}, \
               impl_trait_ref={:?}, skol_obligation_trait_ref={:?})",
               impl_def_id,
               obligation,
               impl_trait_ref,
               skol_obligation_trait_ref);

        let origin = TypeOrigin::RelateOutputImplTypes(obligation.cause.span);
        if let Err(e) = self.infcx.eq_trait_refs(false,
                                                 origin,
                                                 impl_trait_ref.value.clone(),
                                                 skol_obligation_trait_ref) {
            debug!("match_impl: failed eq_trait_refs due to `{}`", e);
            return Err(());
        }

        if let Err(e) = self.infcx.leak_check(&skol_map, snapshot) {
            debug!("match_impl: failed leak check due to `{}`", e);
            return Err(());
        }

        debug!("match_impl: success impl_substs={:?}", impl_substs);
        Ok((Normalized {
            value: impl_substs,
            obligations: impl_trait_ref.obligations
        }, skol_map))
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
            .zip(impl_trait_ref.input_types())
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
        try!(self.match_poly_trait_ref(obligation, where_clause_trait_ref));
        Ok(Vec::new())
    }

    /// Returns `Ok` if `poly_trait_ref` being true implies that the
    /// obligation is satisfied.
    fn match_poly_trait_ref(&self,
                            obligation: &TraitObligation<'tcx>,
                            poly_trait_ref: ty::PolyTraitRef<'tcx>)
                            -> Result<(),()>
    {
        debug!("match_poly_trait_ref: obligation={:?} poly_trait_ref={:?}",
               obligation,
               poly_trait_ref);

        let origin = TypeOrigin::RelateOutputImplTypes(obligation.cause.span);
        match self.infcx.sub_poly_trait_refs(false,
                                             origin,
                                             poly_trait_ref,
                                             obligation.predicate.to_poly_trait_ref()) {
            Ok(()) => Ok(()),
            Err(_) => Err(()),
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Miscellany

    fn match_fresh_trait_refs(&self,
                              previous: &ty::PolyTraitRef<'tcx>,
                              current: &ty::PolyTraitRef<'tcx>)
                              -> bool
    {
        let mut matcher = ty::_match::Match::new(self.tcx());
        matcher.relate(previous, current).is_ok()
    }

    fn push_stack<'o,'s:'o>(&mut self,
                            previous_stack: TraitObligationStackList<'s, 'tcx>,
                            obligation: &'o TraitObligation<'tcx>)
                            -> TraitObligationStack<'o, 'tcx>
    {
        let fresh_trait_ref =
            obligation.predicate.to_poly_trait_ref().fold_with(&mut self.freshener);

        TraitObligationStack {
            obligation: obligation,
            fresh_trait_ref: fresh_trait_ref,
            previous: previous_stack,
        }
    }

    fn closure_trait_ref_unnormalized(&mut self,
                                      obligation: &TraitObligation<'tcx>,
                                      closure_def_id: DefId,
                                      substs: &ty::ClosureSubsts<'tcx>)
                                      -> ty::PolyTraitRef<'tcx>
    {
        let closure_type = self.infcx.closure_type(closure_def_id, substs);
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

    fn closure_trait_ref(&mut self,
                         obligation: &TraitObligation<'tcx>,
                         closure_def_id: DefId,
                         substs: &ty::ClosureSubsts<'tcx>)
                         -> Normalized<'tcx, ty::PolyTraitRef<'tcx>>
    {
        let trait_ref = self.closure_trait_ref_unnormalized(
            obligation, closure_def_id, substs);

        // A closure signature can contain associated types which
        // must be normalized.
        normalize_with_depth(self,
                             obligation.cause.clone(),
                             obligation.recursion_depth+1,
                             &trait_ref)
    }

    /// Returns the obligations that are implied by instantiating an
    /// impl or trait. The obligations are substituted and fully
    /// normalized. This is used when confirming an impl or default
    /// impl.
    fn impl_or_trait_obligations(&mut self,
                                 cause: ObligationCause<'tcx>,
                                 recursion_depth: usize,
                                 def_id: DefId, // of impl or trait
                                 substs: &Substs<'tcx>, // for impl or trait
                                 skol_map: infer::SkolemizationMap,
                                 snapshot: &infer::CombinedSnapshot)
                                 -> Vec<PredicateObligation<'tcx>>
    {
        debug!("impl_or_trait_obligations(def_id={:?})", def_id);
        let tcx = self.tcx();

        // To allow for one-pass evaluation of the nested obligation,
        // each predicate must be preceded by the obligations required
        // to normalize it.
        // for example, if we have:
        //    impl<U: Iterator, V: Iterator<Item=U>> Foo for V where U::Item: Copy
        // the impl will have the following predicates:
        //    <V as Iterator>::Item = U,
        //    U: Iterator, U: Sized,
        //    V: Iterator, V: Sized,
        //    <U as Iterator>::Item: Copy
        // When we substitute, say, `V => IntoIter<u32>, U => $0`, the last
        // obligation will normalize to `<$0 as Iterator>::Item = $1` and
        // `$1: Copy`, so we must ensure the obligations are emitted in
        // that order.
        let predicates = tcx
            .lookup_predicates(def_id)
            .predicates.iter()
            .flat_map(|predicate| {
                let predicate =
                    normalize_with_depth(self, cause.clone(), recursion_depth,
                                         &predicate.subst(tcx, substs));
                predicate.obligations.into_iter().chain(
                    Some(Obligation {
                        cause: cause.clone(),
                        recursion_depth: recursion_depth,
                        predicate: predicate.value
                    }))
            }).collect();
        self.infcx().plug_leaks(skol_map, snapshot, &predicates)
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
                parent_code: Rc::new(obligation.cause.code.clone())
            };
            let derived_code = variant(derived_cause);
            ObligationCause::new(obligation.cause.span, obligation.cause.body_id, derived_code)
        } else {
            obligation.cause.clone()
        }
    }
}

impl<'tcx> SelectionCache<'tcx> {
    pub fn new() -> SelectionCache<'tcx> {
        SelectionCache {
            hashmap: RefCell::new(FnvHashMap())
        }
    }
}

impl<'tcx> EvaluationCache<'tcx> {
    pub fn new() -> EvaluationCache<'tcx> {
        EvaluationCache {
            hashmap: RefCell::new(FnvHashMap())
        }
    }
}

impl<'o,'tcx> TraitObligationStack<'o,'tcx> {
    fn list(&'o self) -> TraitObligationStackList<'o,'tcx> {
        TraitObligationStackList::with(self)
    }

    fn iter(&'o self) -> TraitObligationStackList<'o,'tcx> {
        self.list()
    }
}

#[derive(Copy, Clone)]
struct TraitObligationStackList<'o,'tcx:'o> {
    head: Option<&'o TraitObligationStack<'o,'tcx>>
}

impl<'o,'tcx> TraitObligationStackList<'o,'tcx> {
    fn empty() -> TraitObligationStackList<'o,'tcx> {
        TraitObligationStackList { head: None }
    }

    fn with(r: &'o TraitObligationStack<'o,'tcx>) -> TraitObligationStackList<'o,'tcx> {
        TraitObligationStackList { head: Some(r) }
    }
}

impl<'o,'tcx> Iterator for TraitObligationStackList<'o,'tcx>{
    type Item = &'o TraitObligationStack<'o,'tcx>;

    fn next(&mut self) -> Option<&'o TraitObligationStack<'o,'tcx>> {
        match self.head {
            Some(o) => {
                *self = o.previous;
                Some(o)
            }
            None => None
        }
    }
}

impl<'o,'tcx> fmt::Debug for TraitObligationStack<'o,'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TraitObligationStack({:?})", self.obligation)
    }
}

impl EvaluationResult {
    fn may_apply(&self) -> bool {
        match *self {
            EvaluatedToOk |
            EvaluatedToAmbig |
            EvaluatedToUnknown => true,

            EvaluatedToErr => false
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
