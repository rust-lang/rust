// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Trait Resolution. See doc.rs.

pub use self::SelectionError::*;
pub use self::FulfillmentErrorCode::*;
pub use self::Vtable::*;
pub use self::ObligationCauseCode::*;

use middle::subst;
use middle::ty::{mod, Ty};
use middle::infer::InferCtxt;
use std::rc::Rc;
use std::slice::Items;
use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};

pub use self::fulfill::{FulfillmentContext, RegionObligation};
pub use self::select::SelectionContext;
pub use self::select::SelectionCache;
pub use self::select::{MethodMatchResult, MethodMatched, MethodAmbiguous, MethodDidNotMatch};
pub use self::select::{MethodMatchedData}; // intentionally don't export variants
pub use self::util::elaborate_predicates;
pub use self::util::supertraits;
pub use self::util::Supertraits;
pub use self::util::search_trait_and_supertraits_from_bound;
pub use self::util::transitive_bounds;
pub use self::util::trait_ref_for_builtin_bound;

mod coherence;
mod fulfill;
mod select;
mod util;

/// An `Obligation` represents some trait reference (e.g. `int:Eq`) for
/// which the vtable must be found.  The process of finding a vtable is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for int`) that
/// provides the required vtable, or else finding a bound that is in
/// scope. The eventual result is usually a `Selection` (defined below).
#[deriving(Clone)]
pub struct Obligation<'tcx, T> {
    pub cause: ObligationCause<'tcx>,
    pub recursion_depth: uint,
    pub trait_ref: T,
}

pub type PredicateObligation<'tcx> = Obligation<'tcx, ty::Predicate<'tcx>>;
pub type TraitObligation<'tcx> = Obligation<'tcx, Rc<ty::TraitRef<'tcx>>>;

/// Why did we incur this obligation? Used for error reporting.
#[deriving(Copy, Clone)]
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

#[deriving(Copy, Clone)]
pub enum ObligationCauseCode<'tcx> {
    /// Not well classified or should be obvious from span.
    MiscObligation,

    /// In an impl of trait X for type Y, type Y must
    /// also implement all supertraits of X.
    ItemObligation(ast::DefId),

    /// Obligation incurred due to an object cast.
    ObjectCastObligation(/* Object type */ Ty<'tcx>),

    /// To implement drop, type must be sendable.
    DropTrait,

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
}

pub type Obligations<'tcx, O> = subst::VecPerParamSpace<Obligation<'tcx, O>>;
pub type PredicateObligations<'tcx> = subst::VecPerParamSpace<PredicateObligation<'tcx>>;
pub type TraitObligations<'tcx> = subst::VecPerParamSpace<TraitObligation<'tcx>>;

pub type Selection<'tcx> = Vtable<'tcx, PredicateObligation<'tcx>>;

#[deriving(Clone,Show)]
pub enum SelectionError<'tcx> {
    Unimplemented,
    Overflow,
    OutputTypeParameterMismatch(Rc<ty::TraitRef<'tcx>>, Rc<ty::TraitRef<'tcx>>, ty::type_err<'tcx>),
}

pub struct FulfillmentError<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub code: FulfillmentErrorCode<'tcx>
}

#[deriving(Clone)]
pub enum FulfillmentErrorCode<'tcx> {
    CodeSelectionError(SelectionError<'tcx>),
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
///    param.clone();    // VtableParam(Oblig_1)
///
///    // Case C: A mix of cases A and B.
///    mixed.clone();    // Vtable(Impl_1, [VtableParam(Oblig_1)])
/// }
/// ```
///
/// ### The type parameter `N`
///
/// See explanation on `VtableImplData`.
#[deriving(Show,Clone)]
pub enum Vtable<'tcx, N> {
    /// Vtable identifying a particular impl.
    VtableImpl(VtableImplData<'tcx, N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter.
    VtableParam(VtableParamData<'tcx>),

    /// Successful resolution for a builtin trait.
    VtableBuiltin(VtableBuiltinData<N>),

    /// Vtable automatically generated for an unboxed closure. The def
    /// ID is the ID of the closure expression. This is a `VtableImpl`
    /// in spirit, but the impl is generated by the compiler and does
    /// not appear in the source.
    VtableUnboxedClosure(ast::DefId, subst::Substs<'tcx>),

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
#[deriving(Clone)]
pub struct VtableImplData<'tcx, N> {
    pub impl_def_id: ast::DefId,
    pub substs: subst::Substs<'tcx>,
    pub nested: subst::VecPerParamSpace<N>
}

#[deriving(Show,Clone)]
pub struct VtableBuiltinData<N> {
    pub nested: subst::VecPerParamSpace<N>
}

/// A vtable provided as a parameter by the caller. For example, in a
/// function like `fn foo<T:Eq>(...)`, if the `eq()` method is invoked
/// on an instance of `T`, the vtable would be of type `VtableParam`.
#[deriving(PartialEq,Eq,Clone)]
pub struct VtableParamData<'tcx> {
    // In the above example, this would `Eq`
    pub bound: Rc<ty::TraitRef<'tcx>>,
}

/// True if neither the trait nor self type is local. Note that `impl_def_id` must refer to an impl
/// of a trait, not an inherent impl.
pub fn is_orphan_impl(tcx: &ty::ctxt,
                      impl_def_id: ast::DefId)
                      -> bool
{
    !coherence::impl_is_local(tcx, impl_def_id)
}

/// True if there exist types that satisfy both of the two given impls.
pub fn overlapping_impls(infcx: &InferCtxt,
                         impl1_def_id: ast::DefId,
                         impl2_def_id: ast::DefId)
                         -> bool
{
    coherence::impl_can_satisfy(infcx, impl1_def_id, impl2_def_id) &&
    coherence::impl_can_satisfy(infcx, impl2_def_id, impl1_def_id)
}

/// Creates predicate obligations from the generic bounds.
pub fn predicates_for_generics<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     cause: ObligationCause<'tcx>,
                                     generic_bounds: &ty::GenericBounds<'tcx>)
                                     -> PredicateObligations<'tcx>
{
    util::predicates_for_generics(tcx, cause, 0, generic_bounds)
}

impl<'tcx,O> Obligation<'tcx,O> {
    pub fn new(cause: ObligationCause<'tcx>,
               trait_ref: O)
               -> Obligation<'tcx, O>
    {
        Obligation { cause: cause,
                     recursion_depth: 0,
                     trait_ref: trait_ref }
    }

    pub fn misc(span: Span, body_id: ast::NodeId, trait_ref: O) -> Obligation<'tcx, O> {
        Obligation::new(ObligationCause::misc(span, body_id), trait_ref)
    }

    pub fn with<P>(&self, value: P) -> Obligation<'tcx,P> {
        Obligation { cause: self.cause.clone(),
                     recursion_depth: self.recursion_depth,
                     trait_ref: value }
    }
}

impl<'tcx> Obligation<'tcx,Rc<ty::TraitRef<'tcx>>> {
    pub fn self_ty(&self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
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
    pub fn iter_nested(&self) -> Items<N> {
        match *self {
            VtableImpl(ref i) => i.iter_nested(),
            VtableFnPointer(..) => (&[]).iter(),
            VtableUnboxedClosure(..) => (&[]).iter(),
            VtableParam(_) => (&[]).iter(),
            VtableBuiltin(ref i) => i.iter_nested(),
        }
    }

    pub fn map_nested<M>(&self, op: |&N| -> M) -> Vtable<'tcx, M> {
        match *self {
            VtableImpl(ref i) => VtableImpl(i.map_nested(op)),
            VtableFnPointer(ref sig) => VtableFnPointer((*sig).clone()),
            VtableUnboxedClosure(d, ref s) => VtableUnboxedClosure(d, s.clone()),
            VtableParam(ref p) => VtableParam((*p).clone()),
            VtableBuiltin(ref b) => VtableBuiltin(b.map_nested(op)),
        }
    }

    pub fn map_move_nested<M>(self, op: |N| -> M) -> Vtable<'tcx, M> {
        match self {
            VtableImpl(i) => VtableImpl(i.map_move_nested(op)),
            VtableFnPointer(sig) => VtableFnPointer(sig),
            VtableUnboxedClosure(d, s) => VtableUnboxedClosure(d, s),
            VtableParam(p) => VtableParam(p),
            VtableBuiltin(no) => VtableBuiltin(no.map_move_nested(op)),
        }
    }
}

impl<'tcx, N> VtableImplData<'tcx, N> {
    pub fn iter_nested(&self) -> Items<N> {
        self.nested.iter()
    }

    pub fn map_nested<M>(&self,
                         op: |&N| -> M)
                         -> VtableImplData<'tcx, M>
    {
        VtableImplData {
            impl_def_id: self.impl_def_id,
            substs: self.substs.clone(),
            nested: self.nested.map(op)
        }
    }

    pub fn map_move_nested<M>(self, op: |N| -> M)
                              -> VtableImplData<'tcx, M> {
        let VtableImplData { impl_def_id, substs, nested } = self;
        VtableImplData {
            impl_def_id: impl_def_id,
            substs: substs,
            nested: nested.map_move(op)
        }
    }
}

impl<N> VtableBuiltinData<N> {
    pub fn iter_nested(&self) -> Items<N> {
        self.nested.iter()
    }

    pub fn map_nested<M>(&self,
                         op: |&N| -> M)
                         -> VtableBuiltinData<M>
    {
        VtableBuiltinData {
            nested: self.nested.map(op)
        }
    }

    pub fn map_move_nested<M>(self, op: |N| -> M) -> VtableBuiltinData<M> {
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
        }
    }
}
