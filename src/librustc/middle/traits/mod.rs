// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Trait Resolution. See the Book for more.

pub use self::SelectionError::*;
pub use self::FulfillmentErrorCode::*;
pub use self::Vtable::*;
pub use self::ObligationCauseCode::*;

use middle::subst;
use middle::ty::{self, HasProjectionTypes, Ty};
use middle::ty_fold::TypeFoldable;
use middle::infer::{self, InferCtxt};
use std::slice::Iter;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};
use util::ppaux::{Repr, UserString};

pub use self::error_reporting::report_fulfillment_errors;
pub use self::error_reporting::suggest_new_overflow_limit;
pub use self::coherence::orphan_check;
pub use self::coherence::overlapping_impls;
pub use self::coherence::OrphanCheckErr;
pub use self::fulfill::{FulfillmentContext, RegionObligation};
pub use self::project::MismatchedProjectionTypes;
pub use self::project::normalize;
pub use self::project::Normalized;
pub use self::object_safety::is_object_safe;
pub use self::object_safety::object_safety_violations;
pub use self::object_safety::ObjectSafetyViolation;
pub use self::object_safety::MethodViolationCode;
pub use self::select::SelectionContext;
pub use self::select::SelectionCache;
pub use self::select::{MethodMatchResult, MethodMatched, MethodAmbiguous, MethodDidNotMatch};
pub use self::select::{MethodMatchedData}; // intentionally don't export variants
pub use self::util::elaborate_predicates;
pub use self::util::get_vtable_index_of_object_method;
pub use self::util::trait_ref_for_builtin_bound;
pub use self::util::supertraits;
pub use self::util::Supertraits;
pub use self::util::transitive_bounds;
pub use self::util::upcast;

mod coherence;
mod error_reporting;
mod fulfill;
mod project;
mod object_safety;
mod select;
mod util;

/// An `Obligation` represents some trait reference (e.g. `int:Eq`) for
/// which the vtable must be found.  The process of finding a vtable is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for int`) that
/// provides the required vtable, or else finding a bound that is in
/// scope. The eventual result is usually a `Selection` (defined below).
#[derive(Clone, PartialEq, Eq)]
pub struct Obligation<'tcx, T> {
    pub cause: ObligationCause<'tcx>,
    pub recursion_depth: uint,
    pub predicate: T,
}

pub type PredicateObligation<'tcx> = Obligation<'tcx, ty::Predicate<'tcx>>;
pub type TraitObligation<'tcx> = Obligation<'tcx, ty::PolyTraitPredicate<'tcx>>;

/// Why did we incur this obligation? Used for error reporting.
#[derive(Clone, PartialEq, Eq)]
pub struct ObligationCause<'tcx> {
    pub span: Span,

    // The id of the fn body that triggered this obligation. This is
    // used for region obligations to determine the precise
    // environment in which the region obligation should be evaluated
    // (in particular, closures can add new assumptions). See the
    // field `region_obligations` of the `FulfillmentContext` for more
    // information.
    pub body_id: ast::NodeId,

    pub code: ObligationCauseCode<'tcx>
}

#[derive(Clone, PartialEq, Eq)]
pub enum ObligationCauseCode<'tcx> {
    /// Not well classified or should be obvious from span.
    MiscObligation,

    /// In an impl of trait X for type Y, type Y must
    /// also implement all supertraits of X.
    ItemObligation(ast::DefId),

    /// Obligation incurred due to an object cast.
    ObjectCastObligation(/* Object type */ Ty<'tcx>),

    /// Various cases where expressions must be sized/copy/etc:
    AssignmentLhsSized,        // L = X implies that L is Sized
    StructInitializerSized,    // S { ... } must be Sized
    VariableType(ast::NodeId), // Type of each variable must be Sized
    ReturnType,                // Return type must be Sized
    RepeatVec,                 // [T,..n] --> T must be Copy

    // Captures of variable the given id by a closure (span is the
    // span of the closure)
    ClosureCapture(ast::NodeId, Span, ty::BuiltinBound),

    // Types of fields (other than the last) in a struct must be sized.
    FieldSized,

    // Only Sized types can be made into objects
    ObjectSized,

    // static items must have `Sync` type
    SharedStatic,


    BuiltinDerivedObligation(DerivedObligationCause<'tcx>),

    ImplDerivedObligation(DerivedObligationCause<'tcx>),

    CompareImplMethodObligation,
}

#[derive(Clone, PartialEq, Eq)]
pub struct DerivedObligationCause<'tcx> {
    /// The trait reference of the parent obligation that led to the
    /// current obligation. Note that only trait obligations lead to
    /// derived obligations, so we just store the trait reference here
    /// directly.
    parent_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The parent trait had this cause
    parent_code: Rc<ObligationCauseCode<'tcx>>
}

pub type Obligations<'tcx, O> = subst::VecPerParamSpace<Obligation<'tcx, O>>;
pub type PredicateObligations<'tcx> = subst::VecPerParamSpace<PredicateObligation<'tcx>>;
pub type TraitObligations<'tcx> = subst::VecPerParamSpace<TraitObligation<'tcx>>;

pub type Selection<'tcx> = Vtable<'tcx, PredicateObligation<'tcx>>;

#[derive(Clone,Debug)]
pub enum SelectionError<'tcx> {
    Unimplemented,
    Overflow,
    OutputTypeParameterMismatch(ty::PolyTraitRef<'tcx>,
                                ty::PolyTraitRef<'tcx>,
                                ty::type_err<'tcx>),
}

pub struct FulfillmentError<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub code: FulfillmentErrorCode<'tcx>
}

#[derive(Clone)]
pub enum FulfillmentErrorCode<'tcx> {
    CodeSelectionError(SelectionError<'tcx>),
    CodeProjectionError(MismatchedProjectionTypes<'tcx>),
    CodeAmbiguity,
}

/// When performing resolution, it is typically the case that there
/// can be one of three outcomes:
///
/// - `Ok(Some(r))`: success occurred with result `r`
/// - `Ok(None)`: could not definitely determine anything, usually due
///   to inconclusive type inference.
/// - `Err(e)`: error `e` occurred
pub type SelectionResult<'tcx, T> = Result<Option<T>, SelectionError<'tcx>>;

/// Given the successful resolution of an obligation, the `Vtable`
/// indicates where the vtable comes from. Note that while we call this
/// a "vtable", it does not necessarily indicate dynamic dispatch at
/// runtime. `Vtable` instances just tell the compiler where to find
/// methods, but in generic code those methods are typically statically
/// dispatched -- only when an object is constructed is a `Vtable`
/// instance reified into an actual vtable.
///
/// For example, the vtable may be tied to a specific impl (case A),
/// or it may be relative to some bound that is in scope (case B).
///
///
/// ```
/// impl<T:Clone> Clone<T> for Option<T> { ... } // Impl_1
/// impl<T:Clone> Clone<T> for Box<T> { ... }    // Impl_2
/// impl Clone for int { ... }             // Impl_3
///
/// fn foo<T:Clone>(concrete: Option<Box<int>>,
///                 param: T,
///                 mixed: Option<T>) {
///
///    // Case A: Vtable points at a specific impl. Only possible when
///    // type is concretely known. If the impl itself has bounded
///    // type parameters, Vtable will carry resolutions for those as well:
///    concrete.clone(); // Vtable(Impl_1, [Vtable(Impl_2, [Vtable(Impl_3)])])
///
///    // Case B: Vtable must be provided by caller. This applies when
///    // type is a type parameter.
///    param.clone();    // VtableParam
///
///    // Case C: A mix of cases A and B.
///    mixed.clone();    // Vtable(Impl_1, [VtableParam])
/// }
/// ```
///
/// ### The type parameter `N`
///
/// See explanation on `VtableImplData`.
#[derive(Debug,Clone)]
pub enum Vtable<'tcx, N> {
    /// Vtable identifying a particular impl.
    VtableImpl(VtableImplData<'tcx, N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter. The `Vec<N>` represents the
    /// obligations incurred from normalizing the where-clause (if
    /// any).
    VtableParam(Vec<N>),

    /// Virtual calls through an object
    VtableObject(VtableObjectData<'tcx>),

    /// Successful resolution for a builtin trait.
    VtableBuiltin(VtableBuiltinData<N>),

    /// Vtable automatically generated for a closure. The def ID is the ID
    /// of the closure expression. This is a `VtableImpl` in spirit, but the
    /// impl is generated by the compiler and does not appear in the source.
    VtableClosure(ast::DefId, subst::Substs<'tcx>),

    /// Same as above, but for a fn pointer type with the given signature.
    VtableFnPointer(ty::Ty<'tcx>),
}

/// Identifies a particular impl in the source, along with a set of
/// substitutions from the impl's type/lifetime parameters. The
/// `nested` vector corresponds to the nested obligations attached to
/// the impl's type parameters.
///
/// The type parameter `N` indicates the type used for "nested
/// obligations" that are required by the impl. During type check, this
/// is `Obligation`, as one might expect. During trans, however, this
/// is `()`, because trans only requires a shallow resolution of an
/// impl, and nested obligations are satisfied later.
#[derive(Clone, PartialEq, Eq)]
pub struct VtableImplData<'tcx, N> {
    pub impl_def_id: ast::DefId,
    pub substs: subst::Substs<'tcx>,
    pub nested: subst::VecPerParamSpace<N>
}

#[derive(Debug,Clone)]
pub struct VtableBuiltinData<N> {
    pub nested: subst::VecPerParamSpace<N>
}

/// A vtable for some object-safe trait `Foo` automatically derived
/// for the object type `Foo`.
#[derive(PartialEq,Eq,Clone)]
pub struct VtableObjectData<'tcx> {
    pub object_ty: Ty<'tcx>,
}

/// Creates predicate obligations from the generic bounds.
pub fn predicates_for_generics<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     cause: ObligationCause<'tcx>,
                                     generic_bounds: &ty::InstantiatedPredicates<'tcx>)
                                     -> PredicateObligations<'tcx>
{
    util::predicates_for_generics(tcx, cause, 0, generic_bounds)
}

/// Determines whether the type `ty` is known to meet `bound` and
/// returns true if so. Returns false if `ty` either does not meet
/// `bound` or is not known to meet bound (note that this is
/// conservative towards *no impl*, which is the opposite of the
/// `evaluate` methods).
pub fn evaluate_builtin_bound<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                                       typer: &ty::ClosureTyper<'tcx>,
                                       ty: Ty<'tcx>,
                                       bound: ty::BuiltinBound,
                                       span: Span)
                                       -> SelectionResult<'tcx, ()>
{
    debug!("type_known_to_meet_builtin_bound(ty={}, bound={:?})",
           ty.repr(infcx.tcx),
           bound);

    let mut fulfill_cx = FulfillmentContext::new();

    // We can use a dummy node-id here because we won't pay any mind
    // to region obligations that arise (there shouldn't really be any
    // anyhow).
    let cause = ObligationCause::misc(span, ast::DUMMY_NODE_ID);

    fulfill_cx.register_builtin_bound(infcx, ty, bound, cause);

    // Note: we only assume something is `Copy` if we can
    // *definitively* show that it implements `Copy`. Otherwise,
    // assume it is move; linear is always ok.
    let result = match fulfill_cx.select_all_or_error(infcx, typer) {
        Ok(()) => Ok(Some(())), // Success, we know it implements Copy.
        Err(errors) => {
            // Check if overflow occurred anywhere and propagate that.
            if errors.iter().any(
                |err| match err.code { CodeSelectionError(Overflow) => true, _ => false })
            {
                return Err(Overflow);
            }

            // Otherwise, if there were any hard errors, propagate an
            // arbitrary one of those. If no hard errors at all,
            // report ambiguity.
            let sel_error =
                errors.iter()
                      .filter_map(|err| {
                          match err.code {
                              CodeAmbiguity => None,
                              CodeSelectionError(ref e) => Some(e.clone()),
                              CodeProjectionError(_) => {
                                  infcx.tcx.sess.span_bug(
                                      span,
                                      "projection error while selecting?")
                              }
                          }
                      })
                      .next();
            match sel_error {
                None => { Ok(None) }
                Some(e) => { Err(e) }
            }
        }
    };

    debug!("type_known_to_meet_builtin_bound: ty={} bound={:?} result={:?}",
           ty.repr(infcx.tcx),
           bound,
           result);

    result
}

pub fn type_known_to_meet_builtin_bound<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                                                 typer: &ty::ClosureTyper<'tcx>,
                                                 ty: Ty<'tcx>,
                                                 bound: ty::BuiltinBound,
                                                 span: Span)
                                                 -> bool
{
    match evaluate_builtin_bound(infcx, typer, ty, bound, span) {
        Ok(Some(())) => {
            // definitely impl'd
            true
        }
        Ok(None) => {
            // ambiguous: if coherence check was successful, shouldn't
            // happen, but we might have reported an error and been
            // soldering on, so just treat this like not implemented
            false
        }
        Err(Overflow) => {
            span_err!(infcx.tcx.sess, span, E0285,
                "overflow evaluating whether `{}` is `{}`",
                      ty.user_string(infcx.tcx),
                      bound.user_string(infcx.tcx));
            suggest_new_overflow_limit(infcx.tcx, span);
            false
        }
        Err(_) => {
            // other errors: not implemented.
            false
        }
    }
}

pub fn normalize_param_env_or_error<'a,'tcx>(unnormalized_env: ty::ParameterEnvironment<'a,'tcx>,
                                             cause: ObligationCause<'tcx>)
                                             -> ty::ParameterEnvironment<'a,'tcx>
{
    match normalize_param_env(&unnormalized_env, cause) {
        Ok(p) => p,
        Err(errors) => {
            // I'm not wild about reporting errors here; I'd prefer to
            // have the errors get reported at a defined place (e.g.,
            // during typeck). Instead I have all parameter
            // environments, in effect, going through this function
            // and hence potentially reporting errors. This ensurse of
            // course that we never forget to normalize (the
            // alternative seemed like it would involve a lot of
            // manual invocations of this fn -- and then we'd have to
            // deal with the errors at each of those sites).
            //
            // In any case, in practice, typeck constructs all the
            // parameter environments once for every fn as it goes,
            // and errors will get reported then; so after typeck we
            // can be sure that no errors should occur.
            let infcx = infer::new_infer_ctxt(unnormalized_env.tcx);
            report_fulfillment_errors(&infcx, &errors);

            // Normalized failed? use what they gave us, it's better than nothing.
            unnormalized_env
        }
    }
}

pub fn normalize_param_env<'a,'tcx>(param_env: &ty::ParameterEnvironment<'a,'tcx>,
                                    cause: ObligationCause<'tcx>)
                                    -> Result<ty::ParameterEnvironment<'a,'tcx>,
                                              Vec<FulfillmentError<'tcx>>>
{
    let tcx = param_env.tcx;

    debug!("normalize_param_env(param_env={})",
           param_env.repr(tcx));

    let infcx = infer::new_infer_ctxt(tcx);
    let predicates = try!(fully_normalize(&infcx, param_env, cause, &param_env.caller_bounds));

    debug!("normalize_param_env: predicates={}",
           predicates.repr(tcx));

    Ok(param_env.with_caller_bounds(predicates))
}

pub fn fully_normalize<'a,'tcx,T>(infcx: &InferCtxt<'a,'tcx>,
                                  closure_typer: &ty::ClosureTyper<'tcx>,
                                  cause: ObligationCause<'tcx>,
                                  value: &T)
                                  -> Result<T, Vec<FulfillmentError<'tcx>>>
    where T : TypeFoldable<'tcx> + HasProjectionTypes + Clone + Repr<'tcx>
{
    let tcx = closure_typer.tcx();

    debug!("normalize_param_env(value={})",
           value.repr(tcx));

    let mut selcx = &mut SelectionContext::new(infcx, closure_typer);
    let mut fulfill_cx = FulfillmentContext::new();
    let Normalized { value: normalized_value, obligations } =
        project::normalize(selcx, cause, value);
    debug!("normalize_param_env: normalized_value={} obligations={}",
           normalized_value.repr(tcx),
           obligations.repr(tcx));
    for obligation in obligations {
        fulfill_cx.register_predicate_obligation(selcx.infcx(), obligation);
    }
    try!(fulfill_cx.select_all_or_error(infcx, closure_typer));
    let resolved_value = infcx.resolve_type_vars_if_possible(&normalized_value);
    debug!("normalize_param_env: resolved_value={}",
           resolved_value.repr(tcx));
    Ok(resolved_value)
}

impl<'tcx,O> Obligation<'tcx,O> {
    pub fn new(cause: ObligationCause<'tcx>,
               trait_ref: O)
               -> Obligation<'tcx, O>
    {
        Obligation { cause: cause,
                     recursion_depth: 0,
                     predicate: trait_ref }
    }

    fn with_depth(cause: ObligationCause<'tcx>,
                  recursion_depth: uint,
                  trait_ref: O)
                  -> Obligation<'tcx, O>
    {
        Obligation { cause: cause,
                     recursion_depth: recursion_depth,
                     predicate: trait_ref }
    }

    pub fn misc(span: Span, body_id: ast::NodeId, trait_ref: O) -> Obligation<'tcx, O> {
        Obligation::new(ObligationCause::misc(span, body_id), trait_ref)
    }

    pub fn with<P>(&self, value: P) -> Obligation<'tcx,P> {
        Obligation { cause: self.cause.clone(),
                     recursion_depth: self.recursion_depth,
                     predicate: value }
    }
}

impl<'tcx> ObligationCause<'tcx> {
    pub fn new(span: Span,
               body_id: ast::NodeId,
               code: ObligationCauseCode<'tcx>)
               -> ObligationCause<'tcx> {
        ObligationCause { span: span, body_id: body_id, code: code }
    }

    pub fn misc(span: Span, body_id: ast::NodeId) -> ObligationCause<'tcx> {
        ObligationCause { span: span, body_id: body_id, code: MiscObligation }
    }

    pub fn dummy() -> ObligationCause<'tcx> {
        ObligationCause { span: DUMMY_SP, body_id: 0, code: MiscObligation }
    }
}

impl<'tcx, N> Vtable<'tcx, N> {
    pub fn iter_nested(&self) -> Iter<N> {
        match *self {
            VtableImpl(ref i) => i.iter_nested(),
            VtableFnPointer(..) => (&[]).iter(),
            VtableClosure(..) => (&[]).iter(),
            VtableParam(ref n) => n.iter(),
            VtableObject(_) => (&[]).iter(),
            VtableBuiltin(ref i) => i.iter_nested(),
        }
    }

    pub fn map_nested<M, F>(&self, op: F) -> Vtable<'tcx, M> where F: FnMut(&N) -> M {
        match *self {
            VtableImpl(ref i) => VtableImpl(i.map_nested(op)),
            VtableFnPointer(ref sig) => VtableFnPointer((*sig).clone()),
            VtableClosure(d, ref s) => VtableClosure(d, s.clone()),
            VtableParam(ref n) => VtableParam(n.iter().map(op).collect()),
            VtableObject(ref p) => VtableObject(p.clone()),
            VtableBuiltin(ref b) => VtableBuiltin(b.map_nested(op)),
        }
    }

    pub fn map_move_nested<M, F>(self, op: F) -> Vtable<'tcx, M> where
        F: FnMut(N) -> M,
    {
        match self {
            VtableImpl(i) => VtableImpl(i.map_move_nested(op)),
            VtableFnPointer(sig) => VtableFnPointer(sig),
            VtableClosure(d, s) => VtableClosure(d, s),
            VtableParam(n) => VtableParam(n.into_iter().map(op).collect()),
            VtableObject(p) => VtableObject(p),
            VtableBuiltin(no) => VtableBuiltin(no.map_move_nested(op)),
        }
    }
}

impl<'tcx, N> VtableImplData<'tcx, N> {
    pub fn iter_nested(&self) -> Iter<N> {
        self.nested.iter()
    }

    pub fn map_nested<M, F>(&self, op: F) -> VtableImplData<'tcx, M> where
        F: FnMut(&N) -> M,
    {
        VtableImplData {
            impl_def_id: self.impl_def_id,
            substs: self.substs.clone(),
            nested: self.nested.map(op)
        }
    }

    pub fn map_move_nested<M, F>(self, op: F) -> VtableImplData<'tcx, M> where
        F: FnMut(N) -> M,
    {
        let VtableImplData { impl_def_id, substs, nested } = self;
        VtableImplData {
            impl_def_id: impl_def_id,
            substs: substs,
            nested: nested.map_move(op)
        }
    }
}

impl<N> VtableBuiltinData<N> {
    pub fn iter_nested(&self) -> Iter<N> {
        self.nested.iter()
    }

    pub fn map_nested<M, F>(&self, op: F) -> VtableBuiltinData<M> where F: FnMut(&N) -> M {
        VtableBuiltinData {
            nested: self.nested.map(op)
        }
    }

    pub fn map_move_nested<M, F>(self, op: F) -> VtableBuiltinData<M> where
        F: FnMut(N) -> M,
    {
        VtableBuiltinData {
            nested: self.nested.map_move(op)
        }
    }
}

impl<'tcx> FulfillmentError<'tcx> {
    fn new(obligation: PredicateObligation<'tcx>,
           code: FulfillmentErrorCode<'tcx>)
           -> FulfillmentError<'tcx>
    {
        FulfillmentError { obligation: obligation, code: code }
    }

    pub fn is_overflow(&self) -> bool {
        match self.code {
            CodeAmbiguity => false,
            CodeSelectionError(Overflow) => true,
            CodeSelectionError(_) => false,
            CodeProjectionError(_) => false,
        }
    }
}

impl<'tcx> TraitObligation<'tcx> {
    fn self_ty(&self) -> Ty<'tcx> {
        self.predicate.0.self_ty()
    }
}
