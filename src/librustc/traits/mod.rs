// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Trait Resolution. See README.md for an overview of how this works.

pub use self::SelectionError::*;
pub use self::FulfillmentErrorCode::*;
pub use self::Vtable::*;
pub use self::ObligationCauseCode::*;

use hir;
use hir::def_id::DefId;
use infer::outlives::env::OutlivesEnvironment;
use middle::const_val::ConstEvalErr;
use middle::region;
use ty::subst::Substs;
use ty::{self, AdtKind, Ty, TyCtxt, TypeFoldable, ToPredicate};
use ty::error::{ExpectedFound, TypeError};
use infer::{InferCtxt};

use std::rc::Rc;
use syntax::ast;
use syntax_pos::{Span, DUMMY_SP};

pub use self::coherence::{orphan_check, overlapping_impls, OrphanCheckErr, OverlapResult};
pub use self::fulfill::FulfillmentContext;
pub use self::project::MismatchedProjectionTypes;
pub use self::project::{normalize, normalize_projection_type, Normalized};
pub use self::project::{ProjectionCache, ProjectionCacheSnapshot, Reveal};
pub use self::object_safety::ObjectSafetyViolation;
pub use self::object_safety::MethodViolationCode;
pub use self::on_unimplemented::{OnUnimplementedDirective, OnUnimplementedNote};
pub use self::select::{EvaluationCache, SelectionContext, SelectionCache};
pub use self::select::IntercrateAmbiguityCause;
pub use self::specialize::{OverlapError, specialization_graph, translate_substs};
pub use self::specialize::{SpecializesCache, find_associated_item};
pub use self::util::elaborate_predicates;
pub use self::util::supertraits;
pub use self::util::Supertraits;
pub use self::util::supertrait_def_ids;
pub use self::util::SupertraitDefIds;
pub use self::util::transitive_bounds;

mod coherence;
mod error_reporting;
mod fulfill;
mod project;
mod object_safety;
mod on_unimplemented;
mod select;
mod specialize;
mod structural_impls;
pub mod trans;
mod util;

// Whether to enable bug compatibility with issue #43355
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum IntercrateMode {
    Issue43355,
    Fixed
}

/// An `Obligation` represents some trait reference (e.g. `int:Eq`) for
/// which the vtable must be found.  The process of finding a vtable is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for int`) that
/// provides the required vtable, or else finding a bound that is in
/// scope. The eventual result is usually a `Selection` (defined below).
#[derive(Clone, PartialEq, Eq)]
pub struct Obligation<'tcx, T> {
    pub cause: ObligationCause<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
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

    /// A slice or array is WF only if `T: Sized`
    SliceOrArrayElem,

    /// A tuple is WF only if its middle elements are Sized
    TupleElem,

    /// This is the trait reference from the given projection
    ProjectionWf(ty::ProjectionTy<'tcx>),

    /// In an impl of trait X for type Y, type Y must
    /// also implement all supertraits of X.
    ItemObligation(DefId),

    /// A type like `&'a T` is WF only if `T: 'a`.
    ReferenceOutlivesReferent(Ty<'tcx>),

    /// A type like `Box<Foo<'a> + 'b>` is WF only if `'b: 'a`.
    ObjectTypeBound(Ty<'tcx>, ty::Region<'tcx>),

    /// Obligation incurred due to an object cast.
    ObjectCastObligation(/* Object type */ Ty<'tcx>),

    // Various cases where expressions must be sized/copy/etc:
    /// L = X implies that L is Sized
    AssignmentLhsSized,
    /// (x1, .., xn) must be Sized
    TupleInitializerSized,
    /// S { ... } must be Sized
    StructInitializerSized,
    /// Type of each variable must be Sized
    VariableType(ast::NodeId),
    /// Return type must be Sized
    SizedReturnType,
    /// [T,..n] --> T must be Copy
    RepeatVec,

    /// Types of fields (other than the last) in a struct must be sized.
    FieldSized(AdtKind),

    /// Constant expressions must be sized.
    ConstSized,

    /// static items must have `Sync` type
    SharedStatic,

    BuiltinDerivedObligation(DerivedObligationCause<'tcx>),

    ImplDerivedObligation(DerivedObligationCause<'tcx>),

    /// error derived when matching traits/impls; see ObligationCause for more details
    CompareImplMethodObligation {
        item_name: ast::Name,
        impl_item_def_id: DefId,
        trait_item_def_id: DefId,
    },

    /// Checking that this expression can be assigned where it needs to be
    // FIXME(eddyb) #11161 is the original Expr required?
    ExprAssignable,

    /// Computing common supertype in the arms of a match expression
    MatchExpressionArm { arm_span: Span,
                         source: hir::MatchSource },

    /// Computing common supertype in an if expression
    IfExpression,

    /// Computing common supertype of an if expression with no else counter-part
    IfExpressionWithNoElse,

    /// `where a == b`
    EquatePredicate,

    /// `main` has wrong type
    MainFunctionType,

    /// `start` has wrong type
    StartFunctionType,

    /// intrinsic has wrong type
    IntrinsicType,

    /// method receiver
    MethodReceiver,

    /// `return` with no expression
    ReturnNoExpression,

    /// `return` with an expression
    ReturnType(ast::NodeId),

    /// Block implicit return
    BlockTailExpression(ast::NodeId),
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
    ConstEvalFailure(ConstEvalErr<'tcx>),
}

pub struct FulfillmentError<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub code: FulfillmentErrorCode<'tcx>
}

#[derive(Clone)]
pub enum FulfillmentErrorCode<'tcx> {
    CodeSelectionError(SelectionError<'tcx>),
    CodeProjectionError(MismatchedProjectionTypes<'tcx>),
    CodeSubtypeError(ExpectedFound<Ty<'tcx>>,
                     TypeError<'tcx>), // always comes from a SubtypePredicate
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
#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum Vtable<'tcx, N> {
    /// Vtable identifying a particular impl.
    VtableImpl(VtableImplData<'tcx, N>),

    /// Vtable for auto trait implementations
    /// This carries the information and nested obligations with regards
    /// to an auto implementation for a trait `Trait`. The nested obligations
    /// ensure the trait implementation holds for all the constituent types.
    VtableAutoImpl(VtableAutoImplData<N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter. The `Vec<N>` represents the
    /// obligations incurred from normalizing the where-clause (if
    /// any).
    VtableParam(Vec<N>),

    /// Virtual calls through an object
    VtableObject(VtableObjectData<'tcx, N>),

    /// Successful resolution for a builtin trait.
    VtableBuiltin(VtableBuiltinData<N>),

    /// Vtable automatically generated for a closure. The def ID is the ID
    /// of the closure expression. This is a `VtableImpl` in spirit, but the
    /// impl is generated by the compiler and does not appear in the source.
    VtableClosure(VtableClosureData<'tcx, N>),

    /// Same as above, but for a fn pointer type with the given signature.
    VtableFnPointer(VtableFnPointerData<'tcx, N>),

    /// Vtable automatically generated for a generator
    VtableGenerator(VtableGeneratorData<'tcx, N>),
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
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct VtableImplData<'tcx, N> {
    pub impl_def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
    pub nested: Vec<N>
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct VtableGeneratorData<'tcx, N> {
    pub closure_def_id: DefId,
    pub substs: ty::ClosureSubsts<'tcx>,
    /// Nested obligations. This can be non-empty if the generator
    /// signature contains associated types.
    pub nested: Vec<N>
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct VtableClosureData<'tcx, N> {
    pub closure_def_id: DefId,
    pub substs: ty::ClosureSubsts<'tcx>,
    /// Nested obligations. This can be non-empty if the closure
    /// signature contains associated types.
    pub nested: Vec<N>
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct VtableAutoImplData<N> {
    pub trait_def_id: DefId,
    pub nested: Vec<N>
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct VtableBuiltinData<N> {
    pub nested: Vec<N>
}

/// A vtable for some object-safe trait `Foo` automatically derived
/// for the object type `Foo`.
#[derive(PartialEq, Eq, Clone, RustcEncodable, RustcDecodable)]
pub struct VtableObjectData<'tcx, N> {
    /// `Foo` upcast to the obligation trait. This will be some supertrait of `Foo`.
    pub upcast_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The vtable is formed by concatenating together the method lists of
    /// the base object trait and all supertraits; this is the start of
    /// `upcast_trait_ref`'s methods in that vtable.
    pub vtable_base: usize,

    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct VtableFnPointerData<'tcx, N> {
    pub fn_ty: Ty<'tcx>,
    pub nested: Vec<N>
}

/// Creates predicate obligations from the generic bounds.
pub fn predicates_for_generics<'tcx>(cause: ObligationCause<'tcx>,
                                     param_env: ty::ParamEnv<'tcx>,
                                     generic_bounds: &ty::InstantiatedPredicates<'tcx>)
                                     -> PredicateObligations<'tcx>
{
    util::predicates_for_generics(cause, 0, param_env, generic_bounds)
}

/// Determines whether the type `ty` is known to meet `bound` and
/// returns true if so. Returns false if `ty` either does not meet
/// `bound` or is not known to meet bound (note that this is
/// conservative towards *no impl*, which is the opposite of the
/// `evaluate` methods).
pub fn type_known_to_meet_bound<'a, 'gcx, 'tcx>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                                param_env: ty::ParamEnv<'tcx>,
                                                ty: Ty<'tcx>,
                                                def_id: DefId,
                                                span: Span)
-> bool
{
    debug!("type_known_to_meet_bound(ty={:?}, bound={:?})",
           ty,
           infcx.tcx.item_path_str(def_id));

    let trait_ref = ty::TraitRef {
        def_id,
        substs: infcx.tcx.mk_substs_trait(ty, &[]),
    };
    let obligation = Obligation {
        param_env,
        cause: ObligationCause::misc(span, ast::DUMMY_NODE_ID),
        recursion_depth: 0,
        predicate: trait_ref.to_predicate(),
    };

    let result = SelectionContext::new(infcx)
        .evaluate_obligation_conservatively(&obligation);
    debug!("type_known_to_meet_ty={:?} bound={} => {:?}",
           ty, infcx.tcx.item_path_str(def_id), result);

    if result && (ty.has_infer_types() || ty.has_closure_types()) {
        // Because of inference "guessing", selection can sometimes claim
        // to succeed while the success requires a guess. To ensure
        // this function's result remains infallible, we must confirm
        // that guess. While imperfect, I believe this is sound.

        // The handling of regions in this area of the code is terrible,
        // see issue #29149. We should be able to improve on this with
        // NLL.
        let mut fulfill_cx = FulfillmentContext::new_ignoring_regions();

        // We can use a dummy node-id here because we won't pay any mind
        // to region obligations that arise (there shouldn't really be any
        // anyhow).
        let cause = ObligationCause::misc(span, ast::DUMMY_NODE_ID);

        fulfill_cx.register_bound(infcx, param_env, ty, def_id, cause);

        // Note: we only assume something is `Copy` if we can
        // *definitively* show that it implements `Copy`. Otherwise,
        // assume it is move; linear is always ok.
        match fulfill_cx.select_all_or_error(infcx) {
            Ok(()) => {
                debug!("type_known_to_meet_bound: ty={:?} bound={} success",
                       ty,
                       infcx.tcx.item_path_str(def_id));
                true
            }
            Err(e) => {
                debug!("type_known_to_meet_bound: ty={:?} bound={} errors={:?}",
                       ty,
                       infcx.tcx.item_path_str(def_id),
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
pub fn normalize_param_env_or_error<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                              region_context: DefId,
                                              unnormalized_env: ty::ParamEnv<'tcx>,
                                              cause: ObligationCause<'tcx>)
                                              -> ty::ParamEnv<'tcx>
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

    let span = cause.span;

    debug!("normalize_param_env_or_error(unnormalized_env={:?})",
           unnormalized_env);

    let predicates: Vec<_> =
        util::elaborate_predicates(tcx, unnormalized_env.caller_bounds.to_vec())
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

    let elaborated_env = ty::ParamEnv::new(tcx.intern_predicates(&predicates),
                                           unnormalized_env.reveal);

    tcx.infer_ctxt().enter(|infcx| {
        // FIXME. We should really... do something with these region
        // obligations. But this call just continues the older
        // behavior (i.e., doesn't cause any new bugs), and it would
        // take some further refactoring to actually solve them. In
        // particular, we would have to handle implied bounds
        // properly, and that code is currently largely confined to
        // regionck (though I made some efforts to extract it
        // out). -nmatsakis
        //
        // @arielby: In any case, these obligations are checked
        // by wfcheck anyway, so I'm not sure we have to check
        // them here too, and we will remove this function when
        // we move over to lazy normalization *anyway*.
        let fulfill_cx = FulfillmentContext::new_ignoring_regions();

        let predicates = match fully_normalize_with_fulfillcx(
            &infcx,
            fulfill_cx,
            cause,
            elaborated_env,
            // You would really want to pass infcx.param_env.caller_bounds here,
            // but that is an interned slice, and fully_normalize takes &T and returns T, so
            // without further refactoring, a slice can't be used. Luckily, we still have the
            // predicate vector from which we created the ParamEnv in infcx, so we
            // can pass that instead. It's roundabout and a bit brittle, but this code path
            // ought to be refactored anyway, and until then it saves us from having to copy.
            &predicates,
        ) {
            Ok(predicates) => predicates,
            Err(errors) => {
                infcx.report_fulfillment_errors(&errors, None);
                // An unnormalized env is better than nothing.
                return elaborated_env;
            }
        };

        debug!("normalize_param_env_or_error: normalized predicates={:?}",
            predicates);

        let region_scope_tree = region::ScopeTree::default();

        // We can use the `elaborated_env` here; the region code only
        // cares about declarations like `'a: 'b`.
        let outlives_env = OutlivesEnvironment::new(elaborated_env);

        infcx.resolve_regions_and_report_errors(region_context, &region_scope_tree, &outlives_env);

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
                tcx.sess.span_err(span, &fixup_err.to_string());
                // An unnormalized env is better than nothing.
                return elaborated_env;
            }
        };

        let predicates = match tcx.lift_to_global(&predicates) {
            Some(predicates) => predicates,
            None => return elaborated_env,
        };

        debug!("normalize_param_env_or_error: resolved predicates={:?}",
               predicates);

        ty::ParamEnv::new(tcx.intern_predicates(&predicates), unnormalized_env.reveal)
    })
}

pub fn fully_normalize<'a, 'gcx, 'tcx, T>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                          cause: ObligationCause<'tcx>,
                                          param_env: ty::ParamEnv<'tcx>,
                                          value: &T)
                                          -> Result<T, Vec<FulfillmentError<'tcx>>>
    where T : TypeFoldable<'tcx>
{
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
    let fulfillcx = FulfillmentContext::new();
    fully_normalize_with_fulfillcx(infcx, fulfillcx, cause, param_env, value)
}

pub fn fully_normalize_with_fulfillcx<'a, 'gcx, 'tcx, T>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    mut fulfill_cx: FulfillmentContext<'tcx>,
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    value: &T)
    -> Result<T, Vec<FulfillmentError<'tcx>>>
    where T : TypeFoldable<'tcx>
{
    debug!("fully_normalize_with_fulfillcx(value={:?})", value);
    let selcx = &mut SelectionContext::new(infcx);
    let Normalized { value: normalized_value, obligations } =
        project::normalize(selcx, param_env, cause, value);
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

/// Normalizes the predicates and checks whether they hold in an empty
/// environment. If this returns false, then either normalize
/// encountered an error or one of the predicates did not hold. Used
/// when creating vtables to check for unsatisfiable methods.
fn normalize_and_test_predicates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           predicates: Vec<ty::Predicate<'tcx>>)
                                           -> bool
{
    debug!("normalize_and_test_predicates(predicates={:?})",
           predicates);

    let result = tcx.infer_ctxt().enter(|infcx| {
        let param_env = ty::ParamEnv::empty(Reveal::All);
        let mut selcx = SelectionContext::new(&infcx);
        let mut fulfill_cx = FulfillmentContext::new();
        let cause = ObligationCause::dummy();
        let Normalized { value: predicates, obligations } =
            normalize(&mut selcx, param_env, cause.clone(), &predicates);
        for obligation in obligations {
            fulfill_cx.register_predicate_obligation(&infcx, obligation);
        }
        for predicate in predicates {
            let obligation = Obligation::new(cause.clone(), param_env, predicate);
            fulfill_cx.register_predicate_obligation(&infcx, obligation);
        }

        fulfill_cx.select_all_or_error(&infcx).is_ok()
    });
    debug!("normalize_and_test_predicates(predicates={:?}) = {:?}",
           predicates, result);
    result
}

fn substitute_normalize_and_test_predicates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                      key: (DefId, &'tcx Substs<'tcx>))
                                                      -> bool
{
    use ty::subst::Subst;
    debug!("substitute_normalize_and_test_predicates(key={:?})",
           key);

    let predicates = tcx.predicates_of(key.0).predicates.subst(tcx, key.1);
    let result = normalize_and_test_predicates(tcx, predicates);

    debug!("substitute_normalize_and_test_predicates(key={:?}) = {:?}",
           key, result);
    result
}

/// Given a trait `trait_ref`, iterates the vtable entries
/// that come from `trait_ref`, including its supertraits.
#[inline] // FIXME(#35870) Avoid closures being unexported due to impl Trait.
fn vtable_methods<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>)
    -> Rc<Vec<Option<(DefId, &'tcx Substs<'tcx>)>>>
{
    debug!("vtable_methods({:?})", trait_ref);

    Rc::new(
        supertraits(tcx, trait_ref).flat_map(move |trait_ref| {
            let trait_methods = tcx.associated_items(trait_ref.def_id())
                .filter(|item| item.kind == ty::AssociatedKind::Method);

            // Now list each method's DefId and Substs (for within its trait).
            // If the method can never be called from this object, produce None.
            trait_methods.map(move |trait_method| {
                debug!("vtable_methods: trait_method={:?}", trait_method);
                let def_id = trait_method.def_id;

                // Some methods cannot be called on an object; skip those.
                if !tcx.is_vtable_safe_method(trait_ref.def_id(), &trait_method) {
                    debug!("vtable_methods: not vtable safe");
                    return None;
                }

                // the method may have some early-bound lifetimes, add
                // regions for those
                let substs = Substs::for_item(tcx, def_id,
                                              |_, _| tcx.types.re_erased,
                                              |def, _| trait_ref.substs().type_for_def(def));

                // the trait type may have higher-ranked lifetimes in it;
                // so erase them if they appear, so that we get the type
                // at some particular call site
                let substs = tcx.erase_late_bound_regions_and_normalize(&ty::Binder(substs));

                // It's possible that the method relies on where clauses that
                // do not hold for this particular set of type parameters.
                // Note that this method could then never be called, so we
                // do not want to try and trans it, in that case (see #23435).
                let predicates = tcx.predicates_of(def_id).instantiate_own(tcx, substs);
                if !normalize_and_test_predicates(tcx, predicates.predicates) {
                    debug!("vtable_methods: predicates do not hold");
                    return None;
                }

                Some((def_id, substs))
            })
        }).collect()
    )
}

impl<'tcx,O> Obligation<'tcx,O> {
    pub fn new(cause: ObligationCause<'tcx>,
               param_env: ty::ParamEnv<'tcx>,
               predicate: O)
               -> Obligation<'tcx, O>
    {
        Obligation { cause, param_env, recursion_depth: 0, predicate }
    }

    fn with_depth(cause: ObligationCause<'tcx>,
                  recursion_depth: usize,
                  param_env: ty::ParamEnv<'tcx>,
                  predicate: O)
                  -> Obligation<'tcx, O>
    {
        Obligation { cause, param_env, recursion_depth, predicate }
    }

    pub fn misc(span: Span,
                body_id: ast::NodeId,
                param_env: ty::ParamEnv<'tcx>,
                trait_ref: O)
                -> Obligation<'tcx, O> {
        Obligation::new(ObligationCause::misc(span, body_id), param_env, trait_ref)
    }

    pub fn with<P>(&self, value: P) -> Obligation<'tcx,P> {
        Obligation { cause: self.cause.clone(),
                     param_env: self.param_env,
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
        ObligationCause { span: DUMMY_SP, body_id: ast::CRATE_NODE_ID, code: MiscObligation }
    }
}

impl<'tcx, N> Vtable<'tcx, N> {
    pub fn nested_obligations(self) -> Vec<N> {
        match self {
            VtableImpl(i) => i.nested,
            VtableParam(n) => n,
            VtableBuiltin(i) => i.nested,
            VtableAutoImpl(d) => d.nested,
            VtableClosure(c) => c.nested,
            VtableGenerator(c) => c.nested,
            VtableObject(d) => d.nested,
            VtableFnPointer(d) => d.nested,
        }
    }

    fn nested_obligations_mut(&mut self) -> &mut Vec<N> {
        match self {
            &mut VtableImpl(ref mut i) => &mut i.nested,
            &mut VtableParam(ref mut n) => n,
            &mut VtableBuiltin(ref mut i) => &mut i.nested,
            &mut VtableAutoImpl(ref mut d) => &mut d.nested,
            &mut VtableGenerator(ref mut c) => &mut c.nested,
            &mut VtableClosure(ref mut c) => &mut c.nested,
            &mut VtableObject(ref mut d) => &mut d.nested,
            &mut VtableFnPointer(ref mut d) => &mut d.nested,
        }
    }

    pub fn map<M, F>(self, f: F) -> Vtable<'tcx, M> where F: FnMut(N) -> M {
        match self {
            VtableImpl(i) => VtableImpl(VtableImplData {
                impl_def_id: i.impl_def_id,
                substs: i.substs,
                nested: i.nested.into_iter().map(f).collect(),
            }),
            VtableParam(n) => VtableParam(n.into_iter().map(f).collect()),
            VtableBuiltin(i) => VtableBuiltin(VtableBuiltinData {
                nested: i.nested.into_iter().map(f).collect(),
            }),
            VtableObject(o) => VtableObject(VtableObjectData {
                upcast_trait_ref: o.upcast_trait_ref,
                vtable_base: o.vtable_base,
                nested: o.nested.into_iter().map(f).collect(),
            }),
            VtableAutoImpl(d) => VtableAutoImpl(VtableAutoImplData {
                trait_def_id: d.trait_def_id,
                nested: d.nested.into_iter().map(f).collect(),
            }),
            VtableFnPointer(p) => VtableFnPointer(VtableFnPointerData {
                fn_ty: p.fn_ty,
                nested: p.nested.into_iter().map(f).collect(),
            }),
            VtableGenerator(c) => VtableGenerator(VtableGeneratorData {
                closure_def_id: c.closure_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            }),
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
    fn self_ty(&self) -> ty::Binder<Ty<'tcx>> {
        ty::Binder(self.predicate.skip_binder().self_ty())
    }
}

pub fn provide(providers: &mut ty::maps::Providers) {
    *providers = ty::maps::Providers {
        is_object_safe: object_safety::is_object_safe_provider,
        specialization_graph_of: specialize::specialization_graph_provider,
        specializes: specialize::specializes,
        trans_fulfill_obligation: trans::trans_fulfill_obligation,
        vtable_methods,
        substitute_normalize_and_test_predicates,
        ..*providers
    };
}
