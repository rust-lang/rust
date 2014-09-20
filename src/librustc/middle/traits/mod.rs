// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Trait Resolution. See doc.rs.
 */

use middle::subst;
use middle::ty;
use middle::typeck::infer::InferCtxt;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};
use util::nodemap::DefIdMap;

pub use self::fulfill::FulfillmentContext;
pub use self::select::SelectionContext;
pub use self::util::supertraits;
pub use self::util::transitive_bounds;
pub use self::util::Supertraits;
pub use self::util::search_trait_and_supertraits_from_bound;

mod coherence;
mod fulfill;
mod select;
mod util;

/**
 * An `Obligation` represents some trait reference (e.g. `int:Eq`) for
 * which the vtable must be found.  The process of finding a vtable is
 * called "resolving" the `Obligation`. This process consists of
 * either identifying an `impl` (e.g., `impl Eq for int`) that
 * provides the required vtable, or else finding a bound that is in
 * scope. The eventual result is usually a `Selection` (defined below).
 */
#[deriving(Clone)]
pub struct Obligation {
    pub cause: ObligationCause,
    pub recursion_depth: uint,
    pub trait_ref: Rc<ty::TraitRef>,
}

/**
 * Why did we incur this obligation? Used for error reporting.
 */
#[deriving(Clone)]
pub struct ObligationCause {
    pub span: Span,
    pub code: ObligationCauseCode
}

#[deriving(Clone)]
pub enum ObligationCauseCode {
    /// Not well classified or should be obvious from span.
    MiscObligation,

    /// In an impl of trait X for type Y, type Y must
    /// also implement all supertraits of X.
    ItemObligation(ast::DefId),

    /// Obligation incurred due to an object cast.
    ObjectCastObligation(/* Object type */ ty::t),

    /// Various cases where expressions must be sized/copy/etc:
    AssignmentLhsSized,        // L = X implies that L is Sized
    StructInitializerSized,    // S { ... } must be Sized
    VariableType(ast::NodeId), // Type of each variable must be Sized
    RepeatVec,                 // [T,..n] --> T must be Copy
}

pub static DUMMY_CAUSE: ObligationCause =
    ObligationCause { span: DUMMY_SP,
                      code: MiscObligation };

pub type Obligations = subst::VecPerParamSpace<Obligation>;

pub type Selection = Vtable<Obligation>;

#[deriving(Clone,Show)]
pub enum SelectionError {
    Unimplemented,
    Overflow,
    OutputTypeParameterMismatch(Rc<ty::TraitRef>, ty::type_err)
}

pub struct FulfillmentError {
    pub obligation: Obligation,
    pub code: FulfillmentErrorCode
}

#[deriving(Clone)]
pub enum FulfillmentErrorCode {
    CodeSelectionError(SelectionError),
    CodeAmbiguity,
}

/**
 * When performing resolution, it is typically the case that there
 * can be one of three outcomes:
 *
 * - `Ok(Some(r))`: success occurred with result `r`
 * - `Ok(None)`: could not definitely determine anything, usually due
 *   to inconclusive type inference.
 * - `Err(e)`: error `e` occurred
 */
pub type SelectionResult<T> = Result<Option<T>, SelectionError>;

#[deriving(PartialEq,Eq,Show)]
pub enum EvaluationResult {
    EvaluatedToMatch,
    EvaluatedToAmbiguity,
    EvaluatedToUnmatch
}

/**
 * Given the successful resolution of an obligation, the `Vtable`
 * indicates where the vtable comes from. Note that while we call this
 * a "vtable", it does not necessarily indicate dynamic dispatch at
 * runtime. `Vtable` instances just tell the compiler where to find
 * methods, but in generic code those methods are typically statically
 * dispatched -- only when an object is constructed is a `Vtable`
 * instance reified into an actual vtable.
 *
 * For example, the vtable may be tied to a specific impl (case A),
 * or it may be relative to some bound that is in scope (case B).
 *
 *
 * ```
 * impl<T:Clone> Clone<T> for Option<T> { ... } // Impl_1
 * impl<T:Clone> Clone<T> for Box<T> { ... }    // Impl_2
 * impl Clone for int { ... }             // Impl_3
 *
 * fn foo<T:Clone>(concrete: Option<Box<int>>,
 *                 param: T,
 *                 mixed: Option<T>) {
 *
 *    // Case A: Vtable points at a specific impl. Only possible when
 *    // type is concretely known. If the impl itself has bounded
 *    // type parameters, Vtable will carry resolutions for those as well:
 *    concrete.clone(); // Vtable(Impl_1, [Vtable(Impl_2, [Vtable(Impl_3)])])
 *
 *    // Case B: Vtable must be provided by caller. This applies when
 *    // type is a type parameter.
 *    param.clone();    // VtableParam(Oblig_1)
 *
 *    // Case C: A mix of cases A and B.
 *    mixed.clone();    // Vtable(Impl_1, [VtableParam(Oblig_1)])
 * }
 * ```
 *
 * ### The type parameter `N`
 *
 * See explanation on `VtableImplData`.
 */
#[deriving(Show,Clone)]
pub enum Vtable<N> {
    /// Vtable identifying a particular impl.
    VtableImpl(VtableImplData<N>),

    /// Vtable automatically generated for an unboxed closure. The def
    /// ID is the ID of the closure expression. This is a `VtableImpl`
    /// in spirit, but the impl is generated by the compiler and does
    /// not appear in the source.
    VtableUnboxedClosure(ast::DefId),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter.
    VtableParam(VtableParamData),

    /// Successful resolution for a builtin trait.
    VtableBuiltin,
}

/**
 * Identifies a particular impl in the source, along with a set of
 * substitutions from the impl's type/lifetime parameters. The
 * `nested` vector corresponds to the nested obligations attached to
 * the impl's type parameters.
 *
 * The type parameter `N` indicates the type used for "nested
 * obligations" that are required by the impl. During type check, this
 * is `Obligation`, as one might expect. During trans, however, this
 * is `()`, because trans only requires a shallow resolution of an
 * impl, and nested obligations are satisfied later.
 */
#[deriving(Clone)]
pub struct VtableImplData<N> {
    pub impl_def_id: ast::DefId,
    pub substs: subst::Substs,
    pub nested: subst::VecPerParamSpace<N>
}

/**
 * A vtable provided as a parameter by the caller. For example, in a
 * function like `fn foo<T:Eq>(...)`, if the `eq()` method is invoked
 * on an instance of `T`, the vtable would be of type `VtableParam`.
 */
#[deriving(Clone)]
pub struct VtableParamData {
    // In the above example, this would `Eq`
    pub bound: Rc<ty::TraitRef>,
}

pub fn try_select_obligation(infcx: &InferCtxt,
                             param_env: &ty::ParameterEnvironment,
                             unboxed_closures: &DefIdMap<ty::UnboxedClosure>,
                             obligation: &Obligation)
                             -> SelectionResult<Selection>
{
    /*!
     * Attempts to select the impl/bound/etc for the obligation
     * given. Returns `None` if we are unable to resolve, either
     * because of ambiguity or due to insufficient inference.  Note
     * that selection is a shallow process and hence the result may
     * contain nested obligations that must be resolved. The caller is
     * responsible for ensuring that those get resolved. (But see
     * `try_select_obligation_deep` below.)
     */

    let selcx = select::SelectionContext::new(infcx, param_env, unboxed_closures);
    selcx.select(obligation)
}

pub fn evaluate_obligation(infcx: &InferCtxt,
                           param_env: &ty::ParameterEnvironment,
                           obligation: &Obligation,
                           unboxed_closures: &DefIdMap<ty::UnboxedClosure>)
                           -> EvaluationResult
{
    /*!
     * Attempts to resolve the obligation given. Returns `None` if
     * we are unable to resolve, either because of ambiguity or
     * due to insufficient inference.
     */

    let selcx = select::SelectionContext::new(infcx, param_env,
                                              unboxed_closures);
    selcx.evaluate_obligation(obligation)
}

pub fn evaluate_impl(infcx: &InferCtxt,
                     param_env: &ty::ParameterEnvironment,
                     unboxed_closures: &DefIdMap<ty::UnboxedClosure>,
                     cause: ObligationCause,
                     impl_def_id: ast::DefId,
                     self_ty: ty::t)
                     -> EvaluationResult
{
    /*!
     * Tests whether the impl `impl_def_id` can be applied to the self
     * type `self_ty`. This is similar to "selection", but simpler:
     *
     * - It does not take a full trait-ref as input, so it skips over
     *   the "confirmation" step which would reconcile output type
     *   parameters.
     * - It returns an `EvaluationResult`, which is a tri-value return
     *   (yes/no/unknown).
     */

    let selcx = select::SelectionContext::new(infcx, param_env, unboxed_closures);
    selcx.evaluate_impl(impl_def_id, cause, self_ty)
}

pub fn select_inherent_impl(infcx: &InferCtxt,
                            param_env: &ty::ParameterEnvironment,
                            unboxed_closures: &DefIdMap<ty::UnboxedClosure>,
                            cause: ObligationCause,
                            impl_def_id: ast::DefId,
                            self_ty: ty::t)
                            -> SelectionResult<VtableImplData<Obligation>>
{
    /*!
     * Matches the self type of the inherent impl `impl_def_id`
     * against `self_ty` and returns the resulting resolution.  This
     * routine may modify the surrounding type context (for example,
     * it may unify variables).
     */

    // This routine is only suitable for inherent impls. This is
    // because it does not attempt to unify the output type parameters
    // from the trait ref against the values from the obligation.
    // (These things do not apply to inherent impls, for which there
    // is no trait ref nor obligation.)
    //
    // Matching against non-inherent impls should be done with
    // `try_resolve_obligation()`.
    assert!(ty::impl_trait_ref(infcx.tcx, impl_def_id).is_none());

    let selcx = select::SelectionContext::new(infcx, param_env,
                                              unboxed_closures);
    selcx.select_inherent_impl(impl_def_id, cause, self_ty)
}

pub fn is_orphan_impl(tcx: &ty::ctxt,
                      impl_def_id: ast::DefId)
                      -> bool
{
    /*!
     * True if neither the trait nor self type is local. Note that
     * `impl_def_id` must refer to an impl of a trait, not an inherent
     * impl.
     */

    !coherence::impl_is_local(tcx, impl_def_id)
}

pub fn overlapping_impls(infcx: &InferCtxt,
                         impl1_def_id: ast::DefId,
                         impl2_def_id: ast::DefId)
                         -> bool
{
    /*!
     * True if there exist types that satisfy both of the two given impls.
     */

    coherence::impl_can_satisfy(infcx, impl1_def_id, impl2_def_id) &&
    coherence::impl_can_satisfy(infcx, impl2_def_id, impl1_def_id)
}

pub fn obligations_for_generics(tcx: &ty::ctxt,
                                cause: ObligationCause,
                                generics: &ty::Generics,
                                substs: &subst::Substs)
                                -> subst::VecPerParamSpace<Obligation>
{
    /*!
     * Given generics for an impl like:
     *
     *    impl<A:Foo, B:Bar+Qux> ...
     *
     * and a substs vector like `<A=A0, B=B0>`, yields a result like
     *
     *    [[Foo for A0, Bar for B0, Qux for B0], [], []]
     */

    util::obligations_for_generics(tcx, cause, 0, generics, substs)
}

pub fn obligation_for_builtin_bound(tcx: &ty::ctxt,
                                    cause: ObligationCause,
                                    source_ty: ty::t,
                                    builtin_bound: ty::BuiltinBound)
                                    -> Obligation
{
    util::obligation_for_builtin_bound(tcx, cause, builtin_bound, 0, source_ty)
}

impl Obligation {
    pub fn new(cause: ObligationCause, trait_ref: Rc<ty::TraitRef>) -> Obligation {
        Obligation { cause: cause,
                     recursion_depth: 0,
                     trait_ref: trait_ref }
    }

    pub fn misc(span: Span, trait_ref: Rc<ty::TraitRef>) -> Obligation {
        Obligation::new(ObligationCause::misc(span), trait_ref)
    }

    pub fn self_ty(&self) -> ty::t {
        self.trait_ref.self_ty()
    }
}

impl ObligationCause {
    pub fn new(span: Span, code: ObligationCauseCode) -> ObligationCause {
        ObligationCause { span: span, code: code }
    }

    pub fn misc(span: Span) -> ObligationCause {
        ObligationCause { span: span, code: MiscObligation }
    }
}

impl<N> Vtable<N> {
    pub fn map_nested<M>(&self, op: |&N| -> M) -> Vtable<M> {
        match *self {
            VtableImpl(ref i) => VtableImpl(i.map_nested(op)),
            VtableUnboxedClosure(d) => VtableUnboxedClosure(d),
            VtableParam(ref p) => VtableParam((*p).clone()),
            VtableBuiltin => VtableBuiltin,
        }
    }

    pub fn map_move_nested<M>(self, op: |N| -> M) -> Vtable<M> {
        match self {
            VtableImpl(i) => VtableImpl(i.map_move_nested(op)),
            VtableUnboxedClosure(d) => VtableUnboxedClosure(d),
            VtableParam(p) => VtableParam(p),
            VtableBuiltin => VtableBuiltin,
        }
    }
}

impl<N> VtableImplData<N> {
    pub fn map_nested<M>(&self,
                         op: |&N| -> M)
                         -> VtableImplData<M>
    {
        VtableImplData {
            impl_def_id: self.impl_def_id,
            substs: self.substs.clone(),
            nested: self.nested.map(op)
        }
    }

    pub fn map_move_nested<M>(self, op: |N| -> M) -> VtableImplData<M> {
        let VtableImplData { impl_def_id, substs, nested } = self;
        VtableImplData {
            impl_def_id: impl_def_id,
            substs: substs,
            nested: nested.map_move(op)
        }
    }
}

impl EvaluationResult {
    pub fn potentially_applicable(&self) -> bool {
        match *self {
            EvaluatedToMatch | EvaluatedToAmbiguity => true,
            EvaluatedToUnmatch => false
        }
    }
}

impl FulfillmentError {
    fn new(obligation: Obligation, code: FulfillmentErrorCode)
           -> FulfillmentError
    {
        FulfillmentError { obligation: obligation, code: code }
    }
}
