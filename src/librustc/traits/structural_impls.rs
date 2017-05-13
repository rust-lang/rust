// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use traits;
use traits::project::Normalized;
use ty::{Lift, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};

use std::fmt;
use std::rc::Rc;

// structural impls for the structs in traits

impl<'tcx, T: fmt::Debug> fmt::Debug for Normalized<'tcx, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Normalized({:?},{:?})",
               self.value,
               self.obligations)
    }
}

impl<'tcx> fmt::Debug for traits::RegionObligation<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionObligation(sub_region={:?}, sup_type={:?})",
               self.sub_region,
               self.sup_type)
    }
}
impl<'tcx, O: fmt::Debug> fmt::Debug for traits::Obligation<'tcx, O> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Obligation(predicate={:?},depth={})",
               self.predicate,
               self.recursion_depth)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::Vtable<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::VtableImpl(ref v) =>
                write!(f, "{:?}", v),

            super::VtableDefaultImpl(ref t) =>
                write!(f, "{:?}", t),

            super::VtableClosure(ref d) =>
                write!(f, "{:?}", d),

            super::VtableFnPointer(ref d) =>
                write!(f, "VtableFnPointer({:?})", d),

            super::VtableObject(ref d) =>
                write!(f, "{:?}", d),

            super::VtableParam(ref n) =>
                write!(f, "VtableParam({:?})", n),

            super::VtableBuiltin(ref d) =>
                write!(f, "{:?}", d)
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableImplData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableImpl(impl_def_id={:?}, substs={:?}, nested={:?})",
               self.impl_def_id,
               self.substs,
               self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableClosureData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableClosure(closure_def_id={:?}, substs={:?}, nested={:?})",
               self.closure_def_id,
               self.substs,
               self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableBuiltinData<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableBuiltin(nested={:?})", self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableDefaultImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableDefaultImplData(trait_def_id={:?}, nested={:?})",
               self.trait_def_id,
               self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableObjectData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableObject(upcast={:?}, vtable_base={}, nested={:?})",
               self.upcast_trait_ref,
               self.vtable_base,
               self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableFnPointerData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableFnPointer(fn_ty={:?}, nested={:?})",
               self.fn_ty,
               self.nested)
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FulfillmentError({:?},{:?})",
               self.obligation,
               self.code)
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::CodeSelectionError(ref e) => write!(f, "{:?}", e),
            super::CodeProjectionError(ref e) => write!(f, "{:?}", e),
            super::CodeAmbiguity => write!(f, "Ambiguity")
        }
    }
}

impl<'tcx> fmt::Debug for traits::MismatchedProjectionTypes<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MismatchedProjectionTypes({:?})", self.err)
    }
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'a, 'tcx> Lift<'tcx> for traits::SelectionError<'a> {
    type Lifted = traits::SelectionError<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            super::Unimplemented => Some(super::Unimplemented),
            super::OutputTypeParameterMismatch(a, b, ref err) => {
                tcx.lift(&(a, b)).and_then(|(a, b)| {
                    tcx.lift(err).map(|err| {
                        super::OutputTypeParameterMismatch(a, b, err)
                    })
                })
            }
            super::TraitNotObjectSafe(def_id) => {
                Some(super::TraitNotObjectSafe(def_id))
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::ObligationCauseCode<'a> {
    type Lifted = traits::ObligationCauseCode<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            super::MiscObligation => Some(super::MiscObligation),
            super::SliceOrArrayElem => Some(super::SliceOrArrayElem),
            super::TupleElem => Some(super::TupleElem),
            super::ProjectionWf(proj) => tcx.lift(&proj).map(super::ProjectionWf),
            super::ItemObligation(def_id) => Some(super::ItemObligation(def_id)),
            super::ReferenceOutlivesReferent(ty) => {
                tcx.lift(&ty).map(super::ReferenceOutlivesReferent)
            }
            super::ObjectTypeBound(ty, r) => {
                tcx.lift(&ty).and_then(|ty| {
                    tcx.lift(&r).and_then(|r| {
                        Some(super::ObjectTypeBound(ty, r))
                    })
                })
            }
            super::ObjectCastObligation(ty) => {
                tcx.lift(&ty).map(super::ObjectCastObligation)
            }
            super::AssignmentLhsSized => Some(super::AssignmentLhsSized),
            super::StructInitializerSized => Some(super::StructInitializerSized),
            super::VariableType(id) => Some(super::VariableType(id)),
            super::ReturnType => Some(super::ReturnType),
            super::RepeatVec => Some(super::RepeatVec),
            super::FieldSized => Some(super::FieldSized),
            super::ConstSized => Some(super::ConstSized),
            super::SharedStatic => Some(super::SharedStatic),
            super::BuiltinDerivedObligation(ref cause) => {
                tcx.lift(cause).map(super::BuiltinDerivedObligation)
            }
            super::ImplDerivedObligation(ref cause) => {
                tcx.lift(cause).map(super::ImplDerivedObligation)
            }
            super::CompareImplMethodObligation { item_name,
                                                 impl_item_def_id,
                                                 trait_item_def_id,
                                                 lint_id } => {
                Some(super::CompareImplMethodObligation {
                    item_name: item_name,
                    impl_item_def_id: impl_item_def_id,
                    trait_item_def_id: trait_item_def_id,
                    lint_id: lint_id,
                })
            }
            super::ExprAssignable => {
                Some(super::ExprAssignable)
            }
            super::MatchExpressionArm { arm_span, source } => {
                Some(super::MatchExpressionArm { arm_span: arm_span,
                                                 source: source })
            }
            super::IfExpression => {
                Some(super::IfExpression)
            }
            super::IfExpressionWithNoElse => {
                Some(super::IfExpressionWithNoElse)
            }
            super::EquatePredicate => {
                Some(super::EquatePredicate)
            }
            super::MainFunctionType => {
                Some(super::MainFunctionType)
            }
            super::StartFunctionType => {
                Some(super::StartFunctionType)
            }
            super::IntrinsicType => {
                Some(super::IntrinsicType)
            }
            super::MethodReceiver => {
                Some(super::MethodReceiver)
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::DerivedObligationCause<'a> {
    type Lifted = traits::DerivedObligationCause<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.parent_trait_ref).and_then(|trait_ref| {
            tcx.lift(&*self.parent_code).map(|code| {
                traits::DerivedObligationCause {
                    parent_trait_ref: trait_ref,
                    parent_code: Rc::new(code)
                }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::ObligationCause<'a> {
    type Lifted = traits::ObligationCause<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.code).map(|code| {
            traits::ObligationCause {
                span: self.span,
                body_id: self.body_id,
                code: code,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::DeferredObligation<'a> {
    type Lifted = traits::DeferredObligation<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.predicate).and_then(|predicate| {
            tcx.lift(&self.cause).map(|cause| {
                traits::DeferredObligation {
                    predicate: predicate,
                    cause: cause
                }
            })
        })
    }
}

// For trans only.
impl<'a, 'tcx> Lift<'tcx> for traits::Vtable<'a, ()> {
    type Lifted = traits::Vtable<'tcx, ()>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match self.clone() {
            traits::VtableImpl(traits::VtableImplData {
                impl_def_id,
                substs,
                nested
            }) => {
                tcx.lift(&substs).map(|substs| {
                    traits::VtableImpl(traits::VtableImplData {
                        impl_def_id: impl_def_id,
                        substs: substs,
                        nested: nested
                    })
                })
            }
            traits::VtableDefaultImpl(t) => Some(traits::VtableDefaultImpl(t)),
            traits::VtableClosure(traits::VtableClosureData {
                closure_def_id,
                substs,
                nested
            }) => {
                tcx.lift(&substs).map(|substs| {
                    traits::VtableClosure(traits::VtableClosureData {
                        closure_def_id: closure_def_id,
                        substs: substs,
                        nested: nested
                    })
                })
            }
            traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested }) => {
                tcx.lift(&fn_ty).map(|fn_ty| {
                    traits::VtableFnPointer(traits::VtableFnPointerData {
                        fn_ty: fn_ty,
                        nested: nested,
                    })
                })
            }
            traits::VtableParam(n) => Some(traits::VtableParam(n)),
            traits::VtableBuiltin(d) => Some(traits::VtableBuiltin(d)),
            traits::VtableObject(traits::VtableObjectData {
                upcast_trait_ref,
                vtable_base,
                nested
            }) => {
                tcx.lift(&upcast_trait_ref).map(|trait_ref| {
                    traits::VtableObject(traits::VtableObjectData {
                        upcast_trait_ref: trait_ref,
                        vtable_base: vtable_base,
                        nested: nested
                    })
                })
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.

impl<'tcx, O: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Obligation<'tcx, O>
{
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::Obligation {
            cause: self.cause.clone(),
            recursion_depth: self.recursion_depth,
            predicate: self.predicate.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicate.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableImplData<'tcx, N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::VtableImplData {
            impl_def_id: self.impl_def_id,
            substs: self.substs.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor) || self.nested.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableClosureData<'tcx, N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::VtableClosureData {
            closure_def_id: self.closure_def_id,
            substs: self.substs.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor) || self.nested.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableDefaultImplData<N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::VtableDefaultImplData {
            trait_def_id: self.trait_def_id,
            nested: self.nested.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.nested.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableBuiltinData<N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::VtableBuiltinData {
            nested: self.nested.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.nested.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableObjectData<'tcx, N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::VtableObjectData {
            upcast_trait_ref: self.upcast_trait_ref.fold_with(folder),
            vtable_base: self.vtable_base,
            nested: self.nested.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.upcast_trait_ref.visit_with(visitor) || self.nested.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableFnPointerData<'tcx, N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::VtableFnPointerData {
            fn_ty: self.fn_ty.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.fn_ty.visit_with(visitor) || self.nested.visit_with(visitor)
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Vtable<'tcx, N> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            traits::VtableImpl(ref v) => traits::VtableImpl(v.fold_with(folder)),
            traits::VtableDefaultImpl(ref t) => traits::VtableDefaultImpl(t.fold_with(folder)),
            traits::VtableClosure(ref d) => {
                traits::VtableClosure(d.fold_with(folder))
            }
            traits::VtableFnPointer(ref d) => {
                traits::VtableFnPointer(d.fold_with(folder))
            }
            traits::VtableParam(ref n) => traits::VtableParam(n.fold_with(folder)),
            traits::VtableBuiltin(ref d) => traits::VtableBuiltin(d.fold_with(folder)),
            traits::VtableObject(ref d) => traits::VtableObject(d.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            traits::VtableImpl(ref v) => v.visit_with(visitor),
            traits::VtableDefaultImpl(ref t) => t.visit_with(visitor),
            traits::VtableClosure(ref d) => d.visit_with(visitor),
            traits::VtableFnPointer(ref d) => d.visit_with(visitor),
            traits::VtableParam(ref n) => n.visit_with(visitor),
            traits::VtableBuiltin(ref d) => d.visit_with(visitor),
            traits::VtableObject(ref d) => d.visit_with(visitor),
        }
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Normalized<'tcx, T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        Normalized {
            value: self.value.fold_with(folder),
            obligations: self.obligations.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.value.visit_with(visitor) || self.obligations.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::ObligationCauseCode<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            super::ExprAssignable |
            super::MatchExpressionArm { arm_span: _, source: _ } |
            super::IfExpression |
            super::IfExpressionWithNoElse |
            super::EquatePredicate |
            super::MainFunctionType |
            super::StartFunctionType |
            super::IntrinsicType |
            super::MethodReceiver |
            super::MiscObligation |
            super::SliceOrArrayElem |
            super::TupleElem |
            super::ItemObligation(_) |
            super::AssignmentLhsSized |
            super::StructInitializerSized |
            super::VariableType(_) |
            super::ReturnType |
            super::RepeatVec |
            super::FieldSized |
            super::ConstSized |
            super::SharedStatic |
            super::CompareImplMethodObligation { .. } => self.clone(),

            super::ProjectionWf(proj) => super::ProjectionWf(proj.fold_with(folder)),
            super::ReferenceOutlivesReferent(ty) => {
                super::ReferenceOutlivesReferent(ty.fold_with(folder))
            }
            super::ObjectTypeBound(ty, r) => {
                super::ObjectTypeBound(ty.fold_with(folder), r.fold_with(folder))
            }
            super::ObjectCastObligation(ty) => {
                super::ObjectCastObligation(ty.fold_with(folder))
            }
            super::BuiltinDerivedObligation(ref cause) => {
                super::BuiltinDerivedObligation(cause.fold_with(folder))
            }
            super::ImplDerivedObligation(ref cause) => {
                super::ImplDerivedObligation(cause.fold_with(folder))
            }
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            super::ExprAssignable |
            super::MatchExpressionArm { arm_span: _, source: _ } |
            super::IfExpression |
            super::IfExpressionWithNoElse |
            super::EquatePredicate |
            super::MainFunctionType |
            super::StartFunctionType |
            super::IntrinsicType |
            super::MethodReceiver |
            super::MiscObligation |
            super::SliceOrArrayElem |
            super::TupleElem |
            super::ItemObligation(_) |
            super::AssignmentLhsSized |
            super::StructInitializerSized |
            super::VariableType(_) |
            super::ReturnType |
            super::RepeatVec |
            super::FieldSized |
            super::ConstSized |
            super::SharedStatic |
            super::CompareImplMethodObligation { .. } => false,

            super::ProjectionWf(proj) => proj.visit_with(visitor),
            super::ReferenceOutlivesReferent(ty) => ty.visit_with(visitor),
            super::ObjectTypeBound(ty, r) => ty.visit_with(visitor) || r.visit_with(visitor),
            super::ObjectCastObligation(ty) => ty.visit_with(visitor),
            super::BuiltinDerivedObligation(ref cause) => cause.visit_with(visitor),
            super::ImplDerivedObligation(ref cause) => cause.visit_with(visitor)
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::DerivedObligationCause<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::DerivedObligationCause {
            parent_trait_ref: self.parent_trait_ref.fold_with(folder),
            parent_code: self.parent_code.fold_with(folder)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.parent_trait_ref.visit_with(visitor) || self.parent_code.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::ObligationCause<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::ObligationCause {
            span: self.span,
            body_id: self.body_id,
            code: self.code.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.code.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::DeferredObligation<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        traits::DeferredObligation {
            predicate: self.predicate.fold_with(folder),
            cause: self.cause.fold_with(folder)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicate.visit_with(visitor) || self.cause.visit_with(visitor)
    }
}
