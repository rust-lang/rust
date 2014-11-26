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
use self::Candidate::*;
use self::BuiltinBoundConditions::*;
use self::EvaluationResult::*;

use super::{Obligation, ObligationCause};
use super::{SelectionError, Unimplemented, Overflow,
            OutputTypeParameterMismatch};
use super::{Selection};
use super::{SelectionResult};
use super::{VtableBuiltin, VtableImpl, VtableParam, VtableUnboxedClosure};
use super::{VtableImplData, VtableParamData, VtableBuiltinData};
use super::{util};

use middle::fast_reject;
use middle::mem_categorization::Typer;
use middle::subst::{Subst, Substs, VecPerParamSpace};
use middle::ty::{mod, Ty};
use middle::typeck::infer;
use middle::typeck::infer::{InferCtxt, TypeSkolemizer};
use middle::ty_fold::TypeFoldable;
use std::cell::RefCell;
use std::collections::hash_map::HashMap;
use std::rc::Rc;
use syntax::ast;
use util::common::ErrorReported;
use util::ppaux::Repr;

pub struct SelectionContext<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    param_env: &'cx ty::ParameterEnvironment<'tcx>,
    typer: &'cx (Typer<'tcx>+'cx),

    /// Skolemizer used specifically for skolemizing entries on the
    /// obligation stack. This ensures that all entries on the stack
    /// at one time will have the same set of skolemized entries,
    /// which is important for checking for trait bounds that
    /// recursively require themselves.
    skolemizer: TypeSkolemizer<'cx, 'tcx>,

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
struct ObligationStack<'prev, 'tcx: 'prev> {
    obligation: &'prev Obligation<'tcx>,

    /// Trait ref from `obligation` but skolemized with the
    /// selection-context's skolemizer. Used to check for recursion.
    skol_trait_ref: Rc<ty::TraitRef<'tcx>>,

    previous: Option<&'prev ObligationStack<'prev, 'tcx>>
}

pub struct SelectionCache<'tcx> {
    hashmap: RefCell<HashMap<Rc<ty::TraitRef<'tcx>>,
                             SelectionResult<'tcx, Candidate<'tcx>>>>,
}

pub enum MethodMatchResult {
    MethodMatched(MethodMatchedData),
    MethodAmbiguous(/* list of impls that could apply */ Vec<ast::DefId>),
    MethodDidNotMatch,
}

#[deriving(Show)]
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
/// For selection to suceed, there must be exactly one non-ambiguous
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
#[deriving(PartialEq,Eq,Show,Clone)]
enum Candidate<'tcx> {
    BuiltinCandidate(ty::BuiltinBound),
    ParamCandidate(VtableParamData<'tcx>),
    ImplCandidate(ast::DefId),
    UnboxedClosureCandidate(/* closure */ ast::DefId, Substs<'tcx>),
    ErrorCandidate,
}

struct CandidateSet<'tcx> {
    vec: Vec<Candidate<'tcx>>,
    ambiguous: bool
}

enum BuiltinBoundConditions<'tcx> {
    If(Vec<Ty<'tcx>>),
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
               param_env: &'cx ty::ParameterEnvironment<'tcx>,
               typer: &'cx Typer<'tcx>)
               -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
            param_env: param_env,
            typer: typer,
            skolemizer: infcx.skolemizer(),
            intercrate: false,
        }
    }

    pub fn intercrate(infcx: &'cx InferCtxt<'cx, 'tcx>,
                      param_env: &'cx ty::ParameterEnvironment<'tcx>,
                      typer: &'cx Typer<'tcx>)
                      -> SelectionContext<'cx, 'tcx> {
        SelectionContext {
            infcx: infcx,
            param_env: param_env,
            typer: typer,
            skolemizer: infcx.skolemizer(),
            intercrate: true,
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

    /// Evaluates whether the obligation can be satisfied. Returns an indication of whether the
    /// obligation can be satisfied and, if so, by what means. Never affects surrounding typing
    /// environment.
    pub fn select(&mut self, obligation: &Obligation<'tcx>)
                  -> SelectionResult<'tcx, Selection<'tcx>> {
        debug!("select({})", obligation.repr(self.tcx()));
        assert!(!obligation.trait_ref.has_escaping_regions());

        let stack = self.push_stack(None, obligation);
        match try!(self.candidate_from_obligation(&stack)) {
            None => Ok(None),
            Some(candidate) => Ok(Some(try!(self.confirm_candidate(obligation, candidate)))),
        }
    }

    pub fn select_inherent_impl(&mut self,
                                impl_def_id: ast::DefId,
                                obligation_cause: ObligationCause<'tcx>,
                                obligation_self_ty: Ty<'tcx>)
                                -> SelectionResult<'tcx, VtableImplData<'tcx, Obligation<'tcx>>>
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
    // Tests whether an obligation can be selected or whether an impl
    // can be applied to particular types. It skips the "confirmation"
    // step and hence completely ignores output type parameters.
    //
    // The result is "true" if the obligation *may* hold and "false" if
    // we can be sure it does not.

    /// Evaluates whether the obligation `obligation` can be satisfied (by any means).
    pub fn evaluate_obligation(&mut self,
                               obligation: &Obligation<'tcx>)
                               -> bool
    {
        debug!("evaluate_obligation({})",
               obligation.repr(self.tcx()));
        assert!(!obligation.trait_ref.has_escaping_regions());

        let stack = self.push_stack(None, obligation);
        self.evaluate_stack(&stack).may_apply()
    }

    fn evaluate_builtin_bound_recursively<'o>(&mut self,
                                              bound: ty::BuiltinBound,
                                              previous_stack: &ObligationStack<'o, 'tcx>,
                                              ty: Ty<'tcx>)
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

    fn evaluate_obligation_recursively<'o>(&mut self,
                                           previous_stack: Option<&ObligationStack<'o, 'tcx>>,
                                           obligation: &Obligation<'tcx>)
                                           -> EvaluationResult
    {
        debug!("evaluate_obligation_recursively({})",
               obligation.repr(self.tcx()));

        let stack = self.push_stack(previous_stack.map(|x| x), obligation);

        let result = self.evaluate_stack(&stack);

        debug!("result: {}", result);
        result
    }

    fn evaluate_stack<'o>(&mut self,
                          stack: &ObligationStack<'o, 'tcx>)
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
        // a recurssive evaluation that `$1 : Eq` -- as you can
        // imagine, this is just where we started. To avoid that, we
        // check for unbound variables and return an ambiguous (hence possible)
        // match if we've seen this trait before.
        //
        // This suffices to allow chains like `FnMut` implemented in
        // terms of `Fn` etc, but we could probably make this more
        // precise still.
        let input_types = stack.skol_trait_ref.input_types();
        let unbound_input_types = input_types.iter().any(|&t| ty::type_is_skolemized(t));
        if
            unbound_input_types &&
             (self.intercrate ||
              stack.iter().skip(1).any(
                  |prev| stack.skol_trait_ref.def_id == prev.skol_trait_ref.def_id))
        {
            debug!("evaluate_stack_intracrate({}) --> unbound argument, recursion -->  ambiguous",
                   stack.skol_trait_ref.repr(self.tcx()));
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
            debug!("evaluate_stack_intracrate({}) --> recursive",
                   stack.skol_trait_ref.repr(self.tcx()));
            return EvaluatedToOk;
        }

        match self.candidate_from_obligation(stack) {
            Ok(Some(c)) => self.winnow_candidate(stack, &c),
            Ok(None) => EvaluatedToAmbig,
            Err(_) => EvaluatedToErr,
        }
    }

    /// Evaluates whether the impl with id `impl_def_id` could be applied to the self type
    /// `obligation_self_ty`. This can be used either for trait or inherent impls.
    pub fn evaluate_impl(&mut self,
                         impl_def_id: ast::DefId,
                         obligation: &Obligation<'tcx>)
                         -> bool
    {
        debug!("evaluate_impl(impl_def_id={}, obligation={})",
               impl_def_id.repr(self.tcx()),
               obligation.repr(self.tcx()));

        self.infcx.probe(|| {
            match self.match_impl(impl_def_id, obligation) {
                Ok(substs) => {
                    let vtable_impl = self.vtable_impl(impl_def_id,
                                                       substs,
                                                       obligation.cause,
                                                       obligation.recursion_depth + 1);
                    self.winnow_selection(None, VtableImpl(vtable_impl)).may_apply()
                }
                Err(()) => {
                    false
                }
            }
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // METHOD MATCHING
    //
    // Method matching is a variation on the normal select/evaluation
    // situation.  In this scenario, rather than having a full trait
    // reference to select from, we start with an expression like
    // `receiver.method(...)`. This means that we have `rcvr_ty`, the
    // type of the receiver, and we have a possible trait that
    // supplies `method`. We must determine whether the receiver is
    // applicable, taking into account the transformed self type
    // declared on `method`. We also must consider the possibility
    // that `receiver` can be *coerced* into a suitable type (for
    // example, a receiver type like `&(Any+Send)` might be coerced
    // into a receiver like `&Any` to allow for method dispatch).  See
    // the body of `evaluate_method_obligation()` for more details on
    // the algorithm.

    /// Determine whether a trait-method is applicable to a receiver of
    /// type `rcvr_ty`. *Does not affect the inference state.*
    ///
    /// - `rcvr_ty` -- type of the receiver
    /// - `xform_self_ty` -- transformed self type declared on the method, with `Self`
    ///   to a fresh type variable
    /// - `obligation` -- a reference to the trait where the method is declared, with
    ///   the input types on the trait replaced with fresh type variables
    pub fn evaluate_method_obligation(&mut self,
                                      rcvr_ty: Ty<'tcx>,
                                      xform_self_ty: Ty<'tcx>,
                                      obligation: &Obligation<'tcx>)
                                      -> MethodMatchResult
    {
        // Here is the situation. We have a trait method declared (say) like so:
        //
        //     trait TheTrait {
        //         fn the_method(self: Rc<Self>, ...) { ... }
        //     }
        //
        // And then we have a call looking (say) like this:
        //
        //     let x: Rc<Foo> = ...;
        //     x.the_method()
        //
        // Now we want to decide if `TheTrait` is applicable. As a
        // human, we can see that `TheTrait` is applicable if there is
        // an impl for the type `Foo`. But how does the compiler know
        // what impl to look for, given that our receiver has type
        // `Rc<Foo>`? We need to take the method's self type into
        // account.
        //
        // On entry to this function, we have the following inputs:
        //
        // - `rcvr_ty = Rc<Foo>`
        // - `xform_self_ty = Rc<$0>`
        // - `obligation = $0 as TheTrait`
        //
        // We do the match in two phases. The first is a *precise
        // match*, which means that no coercion is required. This is
        // the preferred way to match. It works by first making
        // `rcvr_ty` a subtype of `xform_self_ty`. This unifies `$0`
        // and `Foo`. We can then evaluate (roughly as normal) the
        // trait reference `Foo as TheTrait`.
        //
        // If this fails, we fallback to a coercive match, described below.

        match self.infcx.probe(|| self.match_method_precise(rcvr_ty, xform_self_ty, obligation)) {
            Ok(()) => { return MethodMatched(PreciseMethodMatch); }
            Err(_) => { }
        }

        // Coercive matches work slightly differently and cannot
        // completely reuse the normal trait matching machinery
        // (though they employ many of the same bits and pieces). To
        // see how it works, let's continue with our previous example,
        // but with the following declarations:
        //
        // ```
        // trait Foo : Bar { .. }
        // trait Bar : Baz { ... }
        // trait Baz { ... }
        // impl TheTrait for Bar {
        //     fn the_method(self: Rc<Bar>, ...) { ... }
        // }
        // ```
        //
        // Now we see that the receiver type `Rc<Foo>` is actually an
        // object type. And in fact the impl we want is an impl on the
        // supertrait `Rc<Bar>`.  The precise matching procedure won't
        // find it, however, because `Rc<Foo>` is not a subtype of
        // `Rc<Bar>` -- it is *coercible* to `Rc<Bar>` (actually, such
        // coercions are not yet implemented, but let's leave that
        // aside for now).
        //
        // To handle this case, we employ a different procedure. Recall
        // that our initial state is as follows:
        //
        // - `rcvr_ty = Rc<Foo>`
        // - `xform_self_ty = Rc<$0>`
        // - `obligation = $0 as TheTrait`
        //
        // We now go through each impl and instantiate all of its type
        // variables, yielding the trait reference that the impl
        // provides. In our example, the impl would provide `Bar as
        // TheTrait`.  Next we (try to) unify the trait reference that
        // the impl provides with the input obligation. This would
        // unify `$0` and `Bar`. Now we can see whether the receiver
        // type (`Rc<Foo>`) is *coercible to* the transformed self
        // type (`Rc<$0> == Rc<Bar>`). In this case, the answer is
        // yes, so the impl is considered a candidate.
        //
        // Note that there is the possibility of ambiguity here, even
        // when all types are known. In our example, this might occur
        // if there was *also* an impl of `TheTrait` for `Baz`. In
        // this case, `Rc<Foo>` would be coercible to both `Rc<Bar>`
        // and `Rc<Baz>`. (Note that it is not a *coherence violation*
        // to have impls for both `Bar` and `Baz`, despite this
        // ambiguity).  In this case, we report an error, listing all
        // the applicable impls.  The user can explicitly "up-coerce"
        // to the type they want.
        //
        // Note that this coercion step only considers actual impls
        // found in the source. This is because all the
        // compiler-provided impls (such as those for unboxed
        // closures) do not have relevant coercions. This simplifies
        // life immensely.

        let mut impls =
            self.assemble_method_candidates_from_impls(rcvr_ty, xform_self_ty, obligation);

        if impls.len() > 1 {
            impls.retain(|&c| self.winnow_method_impl(c, rcvr_ty, xform_self_ty, obligation));
        }

        if impls.len() > 1 {
            return MethodAmbiguous(impls);
        }

        match impls.pop() {
            Some(def_id) => MethodMatched(CoerciveMethodMatch(def_id)),
            None => MethodDidNotMatch
        }
    }

    /// Given the successful result of a method match, this function "confirms" the result, which
    /// basically repeats the various matching operations, but outside of any snapshot so that
    /// their effects are committed into the inference state.
    pub fn confirm_method_match(&mut self,
                                rcvr_ty: Ty<'tcx>,
                                xform_self_ty: Ty<'tcx>,
                                obligation: &Obligation<'tcx>,
                                data: MethodMatchedData)
    {
        let is_ok = match data {
            PreciseMethodMatch => {
                self.match_method_precise(rcvr_ty, xform_self_ty, obligation).is_ok()
            }

            CoerciveMethodMatch(impl_def_id) => {
                self.match_method_coerce(impl_def_id, rcvr_ty, xform_self_ty, obligation).is_ok()
            }
        };

        if !is_ok {
            self.tcx().sess.span_bug(
                obligation.cause.span,
                format!("match not repeatable: {}, {}, {}, {}",
                        rcvr_ty.repr(self.tcx()),
                        xform_self_ty.repr(self.tcx()),
                        obligation.repr(self.tcx()),
                        data)[]);
        }
    }

    /// Implements the *precise method match* procedure described in
    /// `evaluate_method_obligation()`.
    fn match_method_precise(&mut self,
                            rcvr_ty: Ty<'tcx>,
                            xform_self_ty: Ty<'tcx>,
                            obligation: &Obligation<'tcx>)
                            -> Result<(),()>
    {
        self.infcx.commit_if_ok(|| {
            match self.infcx.sub_types(false, infer::RelateSelfType(obligation.cause.span),
                                       rcvr_ty, xform_self_ty) {
                Ok(()) => { }
                Err(_) => { return Err(()); }
            }

            if self.evaluate_obligation(obligation) {
                Ok(())
            } else {
                Err(())
            }
        })
    }

    /// Assembles a list of potentially applicable impls using the *coercive match* procedure
    /// described in `evaluate_method_obligation()`.
    fn assemble_method_candidates_from_impls(&mut self,
                                             rcvr_ty: Ty<'tcx>,
                                             xform_self_ty: Ty<'tcx>,
                                             obligation: &Obligation<'tcx>)
                                             -> Vec<ast::DefId>
    {
        let mut candidates = Vec::new();

        let all_impls = self.all_impls(obligation.trait_ref.def_id);
        for &impl_def_id in all_impls.iter() {
            self.infcx.probe(|| {
                match self.match_method_coerce(impl_def_id, rcvr_ty, xform_self_ty, obligation) {
                    Ok(_) => { candidates.push(impl_def_id); }
                    Err(_) => { }
                }
            });
        }

        candidates
    }

    /// Applies the *coercive match* procedure described in `evaluate_method_obligation()` to a
    /// particular impl.
    fn match_method_coerce(&mut self,
                           impl_def_id: ast::DefId,
                           rcvr_ty: Ty<'tcx>,
                           xform_self_ty: Ty<'tcx>,
                           obligation: &Obligation<'tcx>)
                           -> Result<Substs<'tcx>, ()>
    {
        // This is almost always expected to succeed. It
        // causes the impl's self-type etc to be unified with
        // the type variable that is shared between
        // obligation/xform_self_ty. In our example, after
        // this is done, the type of `xform_self_ty` would
        // change from `Rc<$0>` to `Rc<Foo>` (because $0 is
        // unified with `Foo`).
        let substs = try!(self.match_impl(impl_def_id, obligation));

        // Next, check whether we can coerce. For now we require
        // that the coercion be a no-op.
        let origin = infer::Misc(obligation.cause.span);
        match infer::mk_coercety(self.infcx, true, origin,
                                 rcvr_ty, xform_self_ty) {
            Ok(None) => { /* Fallthrough */ }
            Ok(Some(_)) | Err(_) => { return Err(()); }
        }

        Ok(substs)
    }

    /// A version of `winnow_impl` applicable to coerice method matching.  This is basically the
    /// same as `winnow_impl` but it uses the method matching procedure and is specific to impls.
    fn winnow_method_impl(&mut self,
                          impl_def_id: ast::DefId,
                          rcvr_ty: Ty<'tcx>,
                          xform_self_ty: Ty<'tcx>,
                          obligation: &Obligation<'tcx>)
                          -> bool
    {
        debug!("winnow_method_impl: impl_def_id={} rcvr_ty={} xform_self_ty={} obligation={}",
               impl_def_id.repr(self.tcx()),
               rcvr_ty.repr(self.tcx()),
               xform_self_ty.repr(self.tcx()),
               obligation.repr(self.tcx()));

        self.infcx.probe(|| {
            match self.match_method_coerce(impl_def_id, rcvr_ty, xform_self_ty, obligation) {
                Ok(substs) => {
                    let vtable_impl = self.vtable_impl(impl_def_id,
                                                       substs,
                                                       obligation.cause,
                                                       obligation.recursion_depth + 1);
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
                                     stack: &ObligationStack<'o, 'tcx>)
                                     -> SelectionResult<'tcx, Candidate<'tcx>>
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
        assert!(!stack.obligation.trait_ref.has_escaping_regions());

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

    fn candidate_from_obligation_no_cache<'o>(&mut self,
                                              stack: &ObligationStack<'o, 'tcx>)
                                              -> SelectionResult<'tcx, Candidate<'tcx>>
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
        // report ambiguiuty.
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
        Ok(Some(candidate))
    }

    fn pick_candidate_cache(&self,
                            cache_skol_trait_ref: &Rc<ty::TraitRef<'tcx>>)
                            -> &SelectionCache<'tcx>
    {
        // High-level idea: we have to decide whether to consult the
        // cache that is specific to this scope, or to consult the
        // global cache. We want the cache that is specific to this
        // scope whenever where clauses might affect the result.

        // Avoid using the master cache during coherence and just rely
        // on the local cache. This effectively disables caching
        // during coherence. It is really just a simplification to
        // avoid us having to fear that coherence results "pollute"
        // the master cache. Since coherence executes pretty quickly,
        // it's not worth going to more trouble to increase the
        // hit-rate I don't think.
        if self.intercrate {
            return &self.param_env.selection_cache;
        }

        // If the trait refers to any parameters in scope, then use
        // the cache of the param-environment.
        if
            cache_skol_trait_ref.input_types().iter().any(
                |&t| ty::type_has_self(t) || ty::type_has_params(t))
        {
            return &self.param_env.selection_cache;
        }

        // If the trait refers to unbound type variables, and there
        // are where clauses in scope, then use the local environment.
        // If there are no where clauses in scope, which is a very
        // common case, then we can use the global environment.
        // See the discussion in doc.rs for more details.
        if
            !self.param_env.caller_obligations.is_empty()
            &&
            cache_skol_trait_ref.input_types().iter().any(
                |&t| ty::type_has_ty_infer(t))
        {
            return &self.param_env.selection_cache;
        }

        // Otherwise, we can use the global cache.
        &self.tcx().selection_cache
    }

    fn check_candidate_cache(&mut self,
                             cache_skol_trait_ref: Rc<ty::TraitRef<'tcx>>)
                             -> Option<SelectionResult<'tcx, Candidate<'tcx>>>
    {
        let cache = self.pick_candidate_cache(&cache_skol_trait_ref);
        let hashmap = cache.hashmap.borrow();
        hashmap.get(&cache_skol_trait_ref).map(|c| (*c).clone())
    }

    fn insert_candidate_cache(&mut self,
                              cache_skol_trait_ref: Rc<ty::TraitRef<'tcx>>,
                              candidate: SelectionResult<'tcx, Candidate<'tcx>>)
    {
        let cache = self.pick_candidate_cache(&cache_skol_trait_ref);
        let mut hashmap = cache.hashmap.borrow_mut();
        hashmap.insert(cache_skol_trait_ref, candidate);
    }

    fn assemble_candidates<'o>(&mut self,
                               stack: &ObligationStack<'o, 'tcx>)
                               -> Result<CandidateSet<'tcx>, SelectionError<'tcx>>
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

    /// Given an obligation like `<SomeTrait for T>`, search the obligations that the caller
    /// supplied to find out whether it is listed among them.
    ///
    /// Never affects inference environment.
    fn assemble_candidates_from_caller_bounds(&mut self,
                                              obligation: &Obligation<'tcx>,
                                              candidates: &mut CandidateSet<'tcx>)
                                              -> Result<(),SelectionError<'tcx>>
    {
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

    /// Check for the artificial impl that the compiler will create for an obligation like `X :
    /// FnMut<..>` where `X` is an unboxed closure type.
    ///
    /// Note: the type parameters on an unboxed closure candidate are modeled as *output* type
    /// parameters and hence do not affect whether this trait is a match or not. They will be
    /// unified during the confirmation step.
    fn assemble_unboxed_candidates(&mut self,
                                   obligation: &Obligation<'tcx>,
                                   candidates: &mut CandidateSet<'tcx>)
                                   -> Result<(),SelectionError<'tcx>>
    {
        let tcx = self.tcx();
        let kind = if Some(obligation.trait_ref.def_id) == tcx.lang_items.fn_trait() {
            ty::FnUnboxedClosureKind
        } else if Some(obligation.trait_ref.def_id) == tcx.lang_items.fn_mut_trait() {
            ty::FnMutUnboxedClosureKind
        } else if Some(obligation.trait_ref.def_id) == tcx.lang_items.fn_once_trait() {
            ty::FnOnceUnboxedClosureKind
        } else {
            return Ok(()); // not a fn trait, ignore
        };

        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());
        let (closure_def_id, substs) = match self_ty.sty {
            ty::ty_unboxed_closure(id, _, ref substs) => (id, substs.clone()),
            ty::ty_infer(ty::TyVar(_)) => {
                candidates.ambiguous = true;
                return Ok(());
            }
            _ => { return Ok(()); }
        };

        debug!("assemble_unboxed_candidates: self_ty={} obligation={}",
               self_ty.repr(self.tcx()),
               obligation.repr(self.tcx()));

        let closure_kind = match self.typer.unboxed_closures().borrow().get(&closure_def_id) {
            Some(closure) => closure.kind,
            None => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    format!("No entry for unboxed closure: {}",
                            closure_def_id.repr(self.tcx())).as_slice());
            }
        };

        if closure_kind == kind {
            candidates.vec.push(UnboxedClosureCandidate(closure_def_id, substs.clone()));
        }

        Ok(())
    }

    /// Search for impls that might apply to `obligation`.
    fn assemble_candidates_from_impls(&mut self,
                                      obligation: &Obligation<'tcx>,
                                      candidates: &mut CandidateSet<'tcx>)
                                      -> Result<(), SelectionError<'tcx>>
    {
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

    /// Further evaluate `candidate` to decide whether all type parameters match and whether nested
    /// obligations are met. Returns true if `candidate` remains viable after this further
    /// scrutiny.
    fn winnow_candidate<'o>(&mut self,
                            stack: &ObligationStack<'o, 'tcx>,
                            candidate: &Candidate<'tcx>)
                            -> EvaluationResult
    {
        debug!("winnow_candidate: candidate={}", candidate.repr(self.tcx()));
        self.infcx.probe(|| {
            let candidate = (*candidate).clone();
            match self.confirm_candidate(stack.obligation, candidate) {
                Ok(selection) => self.winnow_selection(Some(stack), selection),
                Err(_) => EvaluatedToErr,
            }
        })
    }

    fn winnow_selection<'o>(&mut self,
                            stack: Option<&ObligationStack<'o, 'tcx>>,
                            selection: Selection<'tcx>)
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

    /// Returns true if `candidate_i` should be dropped in favor of `candidate_j`.
    ///
    /// This is generally true if either:
    /// - candidate i and candidate j are equivalent; or,
    /// - candidate i is a conrete impl and candidate j is a where clause bound,
    ///   and the concrete impl is applicable to the types in the where clause bound.
    ///
    /// The last case refers to cases where there are blanket impls (often conditional
    /// blanket impls) as well as a where clause. This can come down to one of two cases:
    ///
    /// - The impl is truly unconditional (it has no where clauses
    ///   of its own), in which case the where clause is
    ///   unnecessary, because coherence requires that we would
    ///   pick that particular impl anyhow (at least so long as we
    ///   don't have specialization).
    ///
    /// - The impl is conditional, in which case we may not have winnowed it out
    ///   because we don't know if the conditions apply, but the where clause is basically
    ///   telling us taht there is some impl, though not necessarily the one we see.
    ///
    /// In both cases we prefer to take the where clause, which is
    /// essentially harmless.  See issue #18453 for more details of
    /// a case where doing the opposite caused us harm.
    fn candidate_should_be_dropped_in_favor_of<'o>(&mut self,
                                                   stack: &ObligationStack<'o, 'tcx>,
                                                   candidate_i: &Candidate<'tcx>,
                                                   candidate_j: &Candidate<'tcx>)
                                                   -> bool
    {
        match (candidate_i, candidate_j) {
            (&ImplCandidate(impl_def_id), &ParamCandidate(ref vt)) => {
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

    fn assemble_builtin_bound_candidates<'o>(&mut self,
                                             bound: ty::BuiltinBound,
                                             stack: &ObligationStack<'o, 'tcx>,
                                             candidates: &mut CandidateSet<'tcx>)
                                             -> Result<(),SelectionError<'tcx>>
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
                     self_ty: Ty<'tcx>)
                     -> Result<BuiltinBoundConditions<'tcx>,SelectionError<'tcx>>
    {
        let self_ty = self.infcx.shallow_resolve(self_ty);
        return match self_ty.sty {
            ty::ty_infer(ty::IntVar(_)) |
            ty::ty_infer(ty::FloatVar(_)) |
            ty::ty_uint(_) |
            ty::ty_int(_) |
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
                                if c.bounds.builtin_bounds.contains(&bound) {
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
                                if c.bounds.builtin_bounds.contains(&bound) {
                                    Ok(If(Vec::new()))
                                } else {
                                    Err(Unimplemented)
                                }
                            }
                        }
                    }
                }
            }

            ty::ty_trait(box ty::TyTrait { ref principal, bounds }) => {
                match bound {
                    ty::BoundSized => {
                        Err(Unimplemented)
                    }
                    ty::BoundCopy | ty::BoundSync | ty::BoundSend => {
                        if bounds.builtin_bounds.contains(&bound) {
                            Ok(If(Vec::new()))
                        } else {
                            // Recursively check all supertraits to find out if any further
                            // bounds are required and thus we must fulfill.
                            // We have to create a temp trait ref here since TyTraits don't
                            // have actual self type info (which is required for the
                            // supertraits iterator).
                            let tmp_tr = Rc::new(ty::TraitRef {
                                def_id: principal.def_id,
                                substs: principal.substs.with_self_ty(ty::mk_err())
                            });
                            for tr in util::supertraits(self.tcx(), tmp_tr) {
                                let td = ty::lookup_trait_def(self.tcx(), tr.def_id);

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
                Ok(If(tys.clone()))
            }

            ty::ty_unboxed_closure(def_id, _, ref substs) => {
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
                match self.tcx().freevars.borrow().get(&def_id.node) {
                    None => {
                        // No upvars.
                        Ok(If(Vec::new()))
                    }

                    Some(freevars) => {
                        let tys: Vec<Ty> =
                            freevars
                            .iter()
                            .map(|freevar| {
                                let freevar_def_id = freevar.def.def_id();
                                self.typer.node_ty(freevar_def_id.node)
                                    .unwrap_or(ty::mk_err()).subst(self.tcx(), substs)
                            })
                            .collect();
                        Ok(If(tys))
                    }
                }
            }

            ty::ty_struct(def_id, ref substs) => {
                let types: Vec<Ty> =
                    ty::struct_fields(self.tcx(), def_id, substs)
                    .iter()
                    .map(|f| f.mt.ty)
                    .collect();
                nominal(self, bound, def_id, types)
            }

            ty::ty_enum(def_id, ref substs) => {
                let types: Vec<Ty> =
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
                         obligation: &Obligation<'tcx>,
                         candidate: Candidate<'tcx>)
                         -> Result<Selection<'tcx>,SelectionError<'tcx>>
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

            UnboxedClosureCandidate(closure_def_id, substs) => {
                try!(self.confirm_unboxed_closure_candidate(obligation, closure_def_id, &substs));
                Ok(VtableUnboxedClosure(closure_def_id, substs))
            }
        }
    }

    fn confirm_param_candidate(&mut self,
                               obligation: &Obligation<'tcx>,
                               param: VtableParamData<'tcx>)
                               -> Result<VtableParamData<'tcx>,
                                         SelectionError<'tcx>>
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
                                 obligation: &Obligation<'tcx>,
                                 bound: ty::BuiltinBound)
                                 -> Result<VtableBuiltinData<Obligation<'tcx>>,
                                           SelectionError<'tcx>>
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
                           obligation: &Obligation<'tcx>,
                           bound: ty::BuiltinBound,
                           nested: Vec<Ty<'tcx>>)
                           -> VtableBuiltinData<Obligation<'tcx>>
    {
        let obligations = nested.iter().map(|&t| {
            util::obligation_for_builtin_bound(
                self.tcx(),
                obligation.cause,
                bound,
                obligation.recursion_depth + 1,
                t)
        }).collect::<Result<_, _>>();
        let obligations = match obligations {
            Ok(o) => o,
            Err(ErrorReported) => Vec::new()
        };
        let obligations = VecPerParamSpace::new(obligations, Vec::new(),
                                                Vec::new(), Vec::new());
        VtableBuiltinData { nested: obligations }
    }

    fn confirm_impl_candidate(&mut self,
                              obligation: &Obligation<'tcx>,
                              impl_def_id: ast::DefId)
                              -> Result<VtableImplData<'tcx, Obligation<'tcx>>,
                                        SelectionError<'tcx>>
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
                   substs: Substs<'tcx>,
                   cause: ObligationCause<'tcx>,
                   recursion_depth: uint)
                   -> VtableImplData<'tcx, Obligation<'tcx>>
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
                                         obligation: &Obligation<'tcx>,
                                         closure_def_id: ast::DefId,
                                         substs: &Substs<'tcx>)
                                         -> Result<(),SelectionError<'tcx>>
    {
        debug!("confirm_unboxed_closure_candidate({},{},{})",
               obligation.repr(self.tcx()),
               closure_def_id.repr(self.tcx()),
               substs.repr(self.tcx()));

        let closure_type = match self.typer.unboxed_closures().borrow().get(&closure_def_id) {
            Some(closure) => closure.closure_type.clone(),
            None => {
                self.tcx().sess.span_bug(
                    obligation.cause.span,
                    format!("No entry for unboxed closure: {}",
                            closure_def_id.repr(self.tcx())).as_slice());
            }
        };

        let closure_sig = &closure_type.sig;
        let arguments_tuple = closure_sig.inputs[0];
        let substs =
            Substs::new_trait(
                vec![arguments_tuple.subst(self.tcx(), substs),
                     closure_sig.output.unwrap().subst(self.tcx(), substs)],
                vec![],
                vec![],
                obligation.self_ty());
        let trait_ref = Rc::new(ty::TraitRef {
            def_id: obligation.trait_ref.def_id,
            substs: substs,
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
                    obligation: &Obligation<'tcx>)
                    -> Substs<'tcx>
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
                  obligation: &Obligation<'tcx>)
                  -> Result<Substs<'tcx>, ()>
    {
        let impl_trait_ref = ty::impl_trait_ref(self.tcx(),
                                                impl_def_id).unwrap();

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

        match self.match_trait_refs(obligation, impl_trait_ref) {
            Ok(()) => Ok(impl_substs),
            Err(()) => Err(())
        }
    }

    fn fast_reject_trait_refs(&mut self,
                              obligation: &Obligation,
                              impl_trait_ref: &ty::TraitRef)
                              -> bool
    {
        // We can avoid creating type variables and doing the full
        // substitution if we find that any of the input types, when
        // simplified, do not match.

        obligation.trait_ref.input_types().iter()
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

    fn match_trait_refs(&mut self,
                        obligation: &Obligation<'tcx>,
                        trait_ref: Rc<ty::TraitRef<'tcx>>)
                        -> Result<(),()>
    {
        debug!("match_trait_refs: obligation={} trait_ref={}",
               obligation.repr(self.tcx()),
               trait_ref.repr(self.tcx()));

        let origin = infer::RelateOutputImplTypes(obligation.cause.span);
        match self.infcx.sub_trait_refs(false,
                                        origin,
                                        trait_ref,
                                        obligation.trait_ref.clone()) {
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
                           obligation_cause: ObligationCause,
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
                        cause: ObligationCause,

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
    // Confirmation
    //
    // The final step of selection: once we know how an obligation is
    // is resolved, we confirm that selection in order to have
    // side-effects on the typing environment. This step also unifies
    // the output type parameters from the obligation with those found
    // on the impl/bound, which may yield type errors.

    /// Relates the output type parameters from an impl to the
    /// trait.  This may lead to type errors. The confirmation step
    /// is separated from the main match procedure because these
    /// type errors do not cause us to select another impl.
    ///
    /// As an example, consider matching the obligation
    /// `Iterator<char> for Elems<int>` using the following impl:
    ///
    ///    impl<T> Iterator<T> for Elems<T> { ... }
    ///
    /// The match phase will succeed with substitution `T=int`.
    /// The confirm step will then try to unify `int` and `char`
    /// and yield an error.
    fn confirm_impl_vtable(&mut self,
                           impl_def_id: ast::DefId,
                           obligation_cause: ObligationCause<'tcx>,
                           obligation_trait_ref: Rc<ty::TraitRef<'tcx>>,
                           substs: &Substs<'tcx>)
                           -> Result<(), SelectionError<'tcx>>
    {
        let impl_trait_ref = ty::impl_trait_ref(self.tcx(),
                                                impl_def_id).unwrap();
        let impl_trait_ref = impl_trait_ref.subst(self.tcx(),
                                                  substs);
        self.confirm(obligation_cause, obligation_trait_ref, impl_trait_ref)
    }

    /// After we have determined which impl applies, and with what substitutions, there is one last
    /// step. We have to go back and relate the "output" type parameters from the obligation to the
    /// types that are specified in the impl.
    ///
    /// For example, imagine we have:
    ///
    ///     impl<T> Iterator<T> for Vec<T> { ... }
    ///
    /// and our obligation is `Iterator<Foo> for Vec<int>` (note the mismatch in the obligation
    /// types). Up until this step, no error would be reported: the self type is `Vec<int>`, and
    /// that matches `Vec<T>` with the substitution `T=int`. At this stage, we could then go and
    /// check that the type parameters to the `Iterator` trait match. (In terms of the parameters,
    /// the `expected_trait_ref` here would be `Iterator<int> for Vec<int>`, and the
    /// `obligation_trait_ref` would be `Iterator<Foo> for Vec<int>`.
    ///
    /// Note that this checking occurs *after* the impl has selected, because these output type
    /// parameters should not affect the selection of the impl. Therefore, if there is a mismatch,
    /// we report an error to the user.
    fn confirm(&mut self,
               obligation_cause: ObligationCause,
               obligation_trait_ref: Rc<ty::TraitRef<'tcx>>,
               expected_trait_ref: Rc<ty::TraitRef<'tcx>>)
               -> Result<(), SelectionError<'tcx>>
    {
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
                            previous_stack: Option<&'s ObligationStack<'s, 'tcx>>,
                            obligation: &'o Obligation<'tcx>)
                            -> ObligationStack<'o, 'tcx>
    {
        let skol_trait_ref = obligation.trait_ref.fold_with(&mut self.skolemizer);

        ObligationStack {
            obligation: obligation,
            skol_trait_ref: skol_trait_ref,
            previous: previous_stack.map(|p| p), // FIXME variance
        }
    }

    /// Returns set of all impls for a given trait.
    fn all_impls(&self, trait_def_id: ast::DefId) -> Vec<ast::DefId> {
        ty::populate_implementations_for_trait_if_necessary(self.tcx(),
                                                            trait_def_id);
        match self.tcx().trait_impls.borrow().get(&trait_def_id) {
            None => Vec::new(),
            Some(impls) => impls.borrow().clone()
        }
    }

    fn impl_obligations(&self,
                        cause: ObligationCause<'tcx>,
                        recursion_depth: uint,
                        impl_def_id: ast::DefId,
                        impl_substs: &Substs<'tcx>)
                        -> VecPerParamSpace<Obligation<'tcx>>
    {
        let impl_generics = ty::lookup_item_type(self.tcx(), impl_def_id).generics;
        let bounds = impl_generics.to_bounds(self.tcx(), impl_substs);
        util::obligations_for_generics(self.tcx(), cause, recursion_depth,
                                       &bounds, &impl_substs.types)
    }
}

impl<'tcx> Repr<'tcx> for Candidate<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            ErrorCandidate => format!("ErrorCandidate"),
            BuiltinCandidate(b) => format!("BuiltinCandidate({})", b),
            UnboxedClosureCandidate(c, ref s) => {
                format!("MatchedUnboxedClosureCandidate({},{})", c, s.repr(tcx))
            }
            ParamCandidate(ref a) => format!("ParamCandidate({})", a.repr(tcx)),
            ImplCandidate(a) => format!("ImplCandidate({})", a.repr(tcx)),
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

impl<'o, 'tcx> ObligationStack<'o, 'tcx> {
    fn iter(&self) -> Option<&ObligationStack<'o, 'tcx>> {
        Some(self)
    }
}

impl<'o, 'tcx> Iterator<&'o ObligationStack<'o, 'tcx>> for Option<&'o ObligationStack<'o, 'tcx>> {
    fn next(&mut self) -> Option<&'o ObligationStack<'o, 'tcx>> {
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

impl<'o, 'tcx> Repr<'tcx> for ObligationStack<'o, 'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
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

impl MethodMatchResult {
    pub fn may_apply(&self) -> bool {
        match *self {
            MethodMatched(_) => true,
            MethodAmbiguous(_) => true,
            MethodDidNotMatch => false,
        }
    }
}
