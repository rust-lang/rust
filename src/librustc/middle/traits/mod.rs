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

use dep_graph::DepNode;
use middle::def_id::DefId;
use middle::free_region::FreeRegionMap;
use middle::subst;
use middle::ty::{self, Ty, TypeFoldable};
use middle::ty::fast_reject;
use middle::infer::{self, fixup_err_to_string, InferCtxt};

use std::rc::Rc;
use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};

pub use self::error_reporting::TraitErrorKey;
pub use self::error_reporting::report_fulfillment_errors;
pub use self::error_reporting::report_overflow_error;
pub use self::error_reporting::report_selection_error;
pub use self::error_reporting::report_object_safety_error;
pub use self::coherence::orphan_check;
pub use self::coherence::overlapping_impls;
pub use self::coherence::OrphanCheckErr;
pub use self::fulfill::{FulfillmentContext, FulfilledPredicates, RegionObligation};
pub use self::project::MismatchedProjectionTypes;
pub use self::project::normalize;
pub use self::project::Normalized;
pub use self::object_safety::is_object_safe;
pub use self::object_safety::astconv_object_safety_violations;
pub use self::object_safety::object_safety_violations;
pub use self::object_safety::ObjectSafetyViolation;
pub use self::object_safety::MethodViolationCode;
pub use self::object_safety::is_vtable_safe_method;
pub use self::select::EvaluationCache;
pub use self::select::SelectionContext;
pub use self::select::SelectionCache;
pub use self::select::{MethodMatchResult, MethodMatched, MethodAmbiguous, MethodDidNotMatch};
pub use self::select::{MethodMatchedData}; // intentionally don't export variants
pub use self::util::elaborate_predicates;
pub use self::util::get_vtable_index_of_object_method;
pub use self::util::trait_ref_for_builtin_bound;
pub use self::util::predicate_for_trait_def;
pub use self::util::supertraits;
pub use self::util::Supertraits;
pub use self::util::supertrait_def_ids;
pub use self::util::SupertraitDefIds;
pub use self::util::transitive_bounds;
pub use self::util::upcast;

mod coherence;
mod error_reporting;
mod fulfill;
mod project;
mod object_safety;
mod select;
mod structural_impls;
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
    pub recursion_depth: usize,
    pub predicate: T,
}

pub type PredicateObligation<'tcx> = Obligation<'tcx, ty::Predicate<'tcx>>;
pub type TraitObligation<'tcx> = Obligation<'tcx, ty::PolyTraitPredicate<'tcx>>;

/// Why did we incur this obligation? Used for error reporting.
#[derive(Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObligationCauseCode<'tcx> {
    /// Not well classified or should be obvious from span.
    MiscObligation,

    /// This is the trait reference from the given projection
    SliceOrArrayElem,

    /// This is the trait reference from the given projection
    ProjectionWf(ty::ProjectionTy<'tcx>),

    /// In an impl of trait X for type Y, type Y must
    /// also implement all supertraits of X.
    ItemObligation(DefId),

    /// A type like `&'a T` is WF only if `T: 'a`.
    ReferenceOutlivesReferent(Ty<'tcx>),

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

    // static items must have `Sync` type
    SharedStatic,

    BuiltinDerivedObligation(DerivedObligationCause<'tcx>),

    ImplDerivedObligation(DerivedObligationCause<'tcx>),

    CompareImplMethodObligation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedObligationCause<'tcx> {
    /// The trait reference of the parent obligation that led to the
    /// current obligation. Note that only trait obligations lead to
    /// derived obligations, so we just store the trait reference here
    /// directly.
    parent_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The parent trait had this cause
    parent_code: Rc<ObligationCauseCode<'tcx>>
}

pub type Obligations<'tcx, O> = Vec<Obligation<'tcx, O>>;
pub type PredicateObligations<'tcx> = Vec<PredicateObligation<'tcx>>;
pub type TraitObligations<'tcx> = Vec<TraitObligation<'tcx>>;

pub type Selection<'tcx> = Vtable<'tcx, PredicateObligation<'tcx>>;

#[derive(Clone,Debug)]
pub enum SelectionError<'tcx> {
    Unimplemented,
    OutputTypeParameterMismatch(ty::PolyTraitRef<'tcx>,
                                ty::PolyTraitRef<'tcx>,
                                ty::error::TypeError<'tcx>),
    TraitNotObjectSafe(DefId),
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
#[derive(Clone)]
pub enum Vtable<'tcx, N> {
    /// Vtable identifying a particular impl.
    VtableImpl(VtableImplData<'tcx, N>),

    /// Vtable for default trait implementations
    /// This carries the information and nested obligations with regards
    /// to a default implementation for a trait `Trait`. The nested obligations
    /// ensure the trait implementation holds for all the constituent types.
    VtableDefaultImpl(VtableDefaultImplData<N>),

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
    VtableClosure(VtableClosureData<'tcx, N>),

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
    pub impl_def_id: DefId,
    pub substs: subst::Substs<'tcx>,
    pub nested: Vec<N>
}

#[derive(Clone, PartialEq, Eq)]
pub struct VtableClosureData<'tcx, N> {
    pub closure_def_id: DefId,
    pub substs: ty::ClosureSubsts<'tcx>,
    /// Nested obligations. This can be non-empty if the closure
    /// signature contains associated types.
    pub nested: Vec<N>
}

#[derive(Clone)]
pub struct VtableDefaultImplData<N> {
    pub trait_def_id: DefId,
    pub nested: Vec<N>
}

#[derive(Clone)]
pub struct VtableBuiltinData<N> {
    pub nested: Vec<N>
}

/// A vtable for some object-safe trait `Foo` automatically derived
/// for the object type `Foo`.
#[derive(PartialEq,Eq,Clone)]
pub struct VtableObjectData<'tcx> {
    /// `Foo` upcast to the obligation trait. This will be some supertrait of `Foo`.
    pub upcast_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The vtable is formed by concatenating together the method lists of
    /// the base object trait and all supertraits; this is the start of
    /// `upcast_trait_ref`'s methods in that vtable.
    pub vtable_base: usize
}

/// Creates predicate obligations from the generic bounds.
pub fn predicates_for_generics<'tcx>(cause: ObligationCause<'tcx>,
                                     generic_bounds: &ty::InstantiatedPredicates<'tcx>)
                                     -> PredicateObligations<'tcx>
{
    util::predicates_for_generics(cause, 0, generic_bounds)
}

/// Determines whether the type `ty` is known to meet `bound` and
/// returns true if so. Returns false if `ty` either does not meet
/// `bound` or is not known to meet bound (note that this is
/// conservative towards *no impl*, which is the opposite of the
/// `evaluate` methods).
pub fn type_known_to_meet_builtin_bound<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                                                 ty: Ty<'tcx>,
                                                 bound: ty::BuiltinBound,
                                                 span: Span)
                                                 -> bool
{
    debug!("type_known_to_meet_builtin_bound(ty={:?}, bound={:?})",
           ty,
           bound);

    let cause = ObligationCause::misc(span, ast::DUMMY_NODE_ID);
    let obligation =
        util::predicate_for_builtin_bound(infcx.tcx, cause, bound, 0, ty);
    let obligation = match obligation {
        Ok(o) => o,
        Err(..) => return false
    };
    let result = SelectionContext::new(infcx)
        .evaluate_obligation_conservatively(&obligation);
    debug!("type_known_to_meet_builtin_bound: ty={:?} bound={:?} => {:?}",
           ty, bound, result);

    if result && (ty.has_infer_types() || ty.has_closure_types()) {
        // Because of inference "guessing", selection can sometimes claim
        // to succeed while the success requires a guess. To ensure
        // this function's result remains infallible, we must confirm
        // that guess. While imperfect, I believe this is sound.

        let mut fulfill_cx = FulfillmentContext::new();

        // We can use a dummy node-id here because we won't pay any mind
        // to region obligations that arise (there shouldn't really be any
        // anyhow).
        let cause = ObligationCause::misc(span, ast::DUMMY_NODE_ID);

        fulfill_cx.register_builtin_bound(infcx, ty, bound, cause);

        // Note: we only assume something is `Copy` if we can
        // *definitively* show that it implements `Copy`. Otherwise,
        // assume it is move; linear is always ok.
        match fulfill_cx.select_all_or_error(infcx) {
            Ok(()) => {
                debug!("type_known_to_meet_builtin_bound: ty={:?} bound={:?} success",
                       ty,
                       bound);
                true
            }
            Err(e) => {
                debug!("type_known_to_meet_builtin_bound: ty={:?} bound={:?} errors={:?}",
                       ty,
                       bound,
                       e);
                false
            }
        }
    } else {
        result
    }
}

// FIXME: this is gonna need to be removed ...
/// Normalizes the parameter environment, reporting errors if they occur.
pub fn normalize_param_env_or_error<'a,'tcx>(unnormalized_env: ty::ParameterEnvironment<'a,'tcx>,
                                             cause: ObligationCause<'tcx>)
                                             -> ty::ParameterEnvironment<'a,'tcx>
{
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

    let tcx = unnormalized_env.tcx;
    let span = cause.span;
    let body_id = cause.body_id;

    debug!("normalize_param_env_or_error(unnormalized_env={:?})",
           unnormalized_env);

    let predicates: Vec<_> =
        util::elaborate_predicates(tcx, unnormalized_env.caller_bounds.clone())
        .filter(|p| !p.is_global()) // (*)
        .collect();

    // (*) Any predicate like `i32: Trait<u32>` or whatever doesn't
    // need to be in the *environment* to be proven, so screen those
    // out. This is important for the soundness of inter-fn
    // caching. Note though that we should probably check that these
    // predicates hold at the point where the environment is
    // constructed, but I am not currently doing so out of laziness.
    // -nmatsakis

    debug!("normalize_param_env_or_error: elaborated-predicates={:?}",
           predicates);

    let elaborated_env = unnormalized_env.with_caller_bounds(predicates);

    let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, Some(elaborated_env));
    let predicates = match fully_normalize(&infcx,
                                           cause,
                                           &infcx.parameter_environment.caller_bounds) {
        Ok(predicates) => predicates,
        Err(errors) => {
            report_fulfillment_errors(&infcx, &errors);
            return infcx.parameter_environment; // an unnormalized env is better than nothing
        }
    };

    debug!("normalize_param_env_or_error: normalized predicates={:?}",
           predicates);

    let free_regions = FreeRegionMap::new();
    infcx.resolve_regions_and_report_errors(&free_regions, body_id);
    let predicates = match infcx.fully_resolve(&predicates) {
        Ok(predicates) => predicates,
        Err(fixup_err) => {
            // If we encounter a fixup error, it means that some type
            // variable wound up unconstrained. I actually don't know
            // if this can happen, and I certainly don't expect it to
            // happen often, but if it did happen it probably
            // represents a legitimate failure due to some kind of
            // unconstrained variable, and it seems better not to ICE,
            // all things considered.
            let err_msg = fixup_err_to_string(fixup_err);
            tcx.sess.span_err(span, &err_msg);
            return infcx.parameter_environment; // an unnormalized env is better than nothing
        }
    };

    debug!("normalize_param_env_or_error: resolved predicates={:?}",
           predicates);

    infcx.parameter_environment.with_caller_bounds(predicates)
}

pub fn fully_normalize<'a,'tcx,T>(infcx: &InferCtxt<'a,'tcx>,
                                  cause: ObligationCause<'tcx>,
                                  value: &T)
                                  -> Result<T, Vec<FulfillmentError<'tcx>>>
    where T : TypeFoldable<'tcx>
{
    debug!("fully_normalize(value={:?})", value);

    let mut selcx = &mut SelectionContext::new(infcx);
    // FIXME (@jroesch) ISSUE 26721
    // I'm not sure if this is a bug or not, needs further investigation.
    // It appears that by reusing the fulfillment_cx here we incur more
    // obligations and later trip an asssertion on regionck.rs line 337.
    //
    // The two possibilities I see is:
    //      - normalization is not actually fully happening and we
    //        have a bug else where
    //      - we are adding a duplicate bound into the list causing
    //        its size to change.
    //
    // I think we should probably land this refactor and then come
    // back to this is a follow-up patch.
    let mut fulfill_cx = FulfillmentContext::new();

    let Normalized { value: normalized_value, obligations } =
        project::normalize(selcx, cause, value);
    debug!("fully_normalize: normalized_value={:?} obligations={:?}",
           normalized_value,
           obligations);
    for obligation in obligations {
        fulfill_cx.register_predicate_obligation(selcx.infcx(), obligation);
    }

    debug!("fully_normalize: select_all_or_error start");
    match fulfill_cx.select_all_or_error(infcx) {
        Ok(()) => { }
        Err(e) => {
            debug!("fully_normalize: error={:?}", e);
            return Err(e);
        }
    }
    debug!("fully_normalize: select_all_or_error complete");
    let resolved_value = infcx.resolve_type_vars_if_possible(&normalized_value);
    debug!("fully_normalize: resolved_value={:?}", resolved_value);
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
                  recursion_depth: usize,
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
    pub fn nested_obligations(self) -> Vec<N> {
        match self {
            VtableImpl(i) => i.nested,
            VtableParam(n) => n,
            VtableBuiltin(i) => i.nested,
            VtableDefaultImpl(d) => d.nested,
            VtableClosure(c) => c.nested,
            VtableObject(_) | VtableFnPointer(..) => vec![]
        }
    }

    pub fn map<M, F>(self, f: F) -> Vtable<'tcx, M> where F: FnMut(N) -> M {
        match self {
            VtableImpl(i) => VtableImpl(VtableImplData {
                impl_def_id: i.impl_def_id,
                substs: i.substs,
                nested: i.nested.into_iter().map(f).collect()
            }),
            VtableParam(n) => VtableParam(n.into_iter().map(f).collect()),
            VtableBuiltin(i) => VtableBuiltin(VtableBuiltinData {
                nested: i.nested.into_iter().map(f).collect()
            }),
            VtableObject(o) => VtableObject(o),
            VtableDefaultImpl(d) => VtableDefaultImpl(VtableDefaultImplData {
                trait_def_id: d.trait_def_id,
                nested: d.nested.into_iter().map(f).collect()
            }),
            VtableFnPointer(f) => VtableFnPointer(f),
            VtableClosure(c) => VtableClosure(VtableClosureData {
                closure_def_id: c.closure_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            })
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
}

impl<'tcx> TraitObligation<'tcx> {
    /// Creates the dep-node for selecting/evaluating this trait reference.
    fn dep_node(&self, tcx: &ty::ctxt<'tcx>) -> DepNode {
        let simplified_ty =
            fast_reject::simplify_type(tcx,
                                       self.predicate.skip_binder().self_ty(), // (*)
                                       true);

        // (*) skip_binder is ok because `simplify_type` doesn't care about regions

        DepNode::TraitSelect(self.predicate.def_id(), simplified_ty)
    }

    fn self_ty(&self) -> ty::Binder<Ty<'tcx>> {
        ty::Binder(self.predicate.skip_binder().self_ty())
    }
}
