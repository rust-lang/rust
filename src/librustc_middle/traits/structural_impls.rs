use crate::traits;
use crate::ty::{Lift, TyCtxt};

use std::fmt;
use std::rc::Rc;

// Structural impls for the structs in `traits`.

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::Vtable<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::VtableImpl(ref v) => write!(f, "{:?}", v),

            super::VtableAutoImpl(ref t) => write!(f, "{:?}", t),

            super::VtableClosure(ref d) => write!(f, "{:?}", d),

            super::VtableGenerator(ref d) => write!(f, "{:?}", d),

            super::VtableFnPointer(ref d) => write!(f, "VtableFnPointer({:?})", d),

            super::VtableDiscriminantKind(ref d) => write!(f, "{:?}", d),

            super::VtableObject(ref d) => write!(f, "{:?}", d),

            super::VtableParam(ref n) => write!(f, "VtableParam({:?})", n),

            super::VtableBuiltin(ref d) => write!(f, "{:?}", d),

            super::VtableTraitAlias(ref d) => write!(f, "{:?}", d),
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableImplData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VtableImplData(impl_def_id={:?}, substs={:?}, nested={:?})",
            self.impl_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableGeneratorData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VtableGeneratorData(generator_def_id={:?}, substs={:?}, nested={:?})",
            self.generator_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableClosureData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VtableClosureData(closure_def_id={:?}, substs={:?}, nested={:?})",
            self.closure_def_id, self.substs, self.nested
        )
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::VtableBuiltinData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VtableBuiltinData(nested={:?})", self.nested)
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::VtableAutoImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VtableAutoImplData(trait_def_id={:?}, nested={:?})",
            self.trait_def_id, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableObjectData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VtableObjectData(upcast={:?}, vtable_base={}, nested={:?})",
            self.upcast_trait_ref, self.vtable_base, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableFnPointerData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VtableFnPointerData(fn_ty={:?}, nested={:?})", self.fn_ty, self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableTraitAliasData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VtableTraitAlias(alias_def_id={:?}, substs={:?}, nested={:?})",
            self.alias_def_id, self.substs, self.nested
        )
    }
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'a, 'tcx> Lift<'tcx> for traits::SelectionError<'a> {
    type Lifted = traits::SelectionError<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            super::Unimplemented => Some(super::Unimplemented),
            super::OutputTypeParameterMismatch(a, b, ref err) => {
                tcx.lift(&(a, b)).and_then(|(a, b)| {
                    tcx.lift(err).map(|err| super::OutputTypeParameterMismatch(a, b, err))
                })
            }
            super::TraitNotObjectSafe(def_id) => Some(super::TraitNotObjectSafe(def_id)),
            super::ConstEvalFailure(err) => Some(super::ConstEvalFailure(err)),
            super::Overflow => Some(super::Overflow),
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::ObligationCauseCode<'a> {
    type Lifted = traits::ObligationCauseCode<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            super::ReturnNoExpression => Some(super::ReturnNoExpression),
            super::MiscObligation => Some(super::MiscObligation),
            super::SliceOrArrayElem => Some(super::SliceOrArrayElem),
            super::TupleElem => Some(super::TupleElem),
            super::ProjectionWf(proj) => tcx.lift(&proj).map(super::ProjectionWf),
            super::ItemObligation(def_id) => Some(super::ItemObligation(def_id)),
            super::BindingObligation(def_id, span) => Some(super::BindingObligation(def_id, span)),
            super::ReferenceOutlivesReferent(ty) => {
                tcx.lift(&ty).map(super::ReferenceOutlivesReferent)
            }
            super::ObjectTypeBound(ty, r) => {
                tcx.lift(&ty).and_then(|ty| tcx.lift(&r).map(|r| super::ObjectTypeBound(ty, r)))
            }
            super::ObjectCastObligation(ty) => tcx.lift(&ty).map(super::ObjectCastObligation),
            super::Coercion { source, target } => {
                Some(super::Coercion { source: tcx.lift(&source)?, target: tcx.lift(&target)? })
            }
            super::AssignmentLhsSized => Some(super::AssignmentLhsSized),
            super::TupleInitializerSized => Some(super::TupleInitializerSized),
            super::StructInitializerSized => Some(super::StructInitializerSized),
            super::VariableType(id) => Some(super::VariableType(id)),
            super::ReturnValue(id) => Some(super::ReturnValue(id)),
            super::ReturnType => Some(super::ReturnType),
            super::SizedArgumentType => Some(super::SizedArgumentType),
            super::SizedReturnType => Some(super::SizedReturnType),
            super::SizedYieldType => Some(super::SizedYieldType),
            super::InlineAsmSized => Some(super::InlineAsmSized),
            super::RepeatVec(suggest_flag) => Some(super::RepeatVec(suggest_flag)),
            super::FieldSized { adt_kind, last } => Some(super::FieldSized { adt_kind, last }),
            super::ConstSized => Some(super::ConstSized),
            super::ConstPatternStructural => Some(super::ConstPatternStructural),
            super::SharedStatic => Some(super::SharedStatic),
            super::BuiltinDerivedObligation(ref cause) => {
                tcx.lift(cause).map(super::BuiltinDerivedObligation)
            }
            super::ImplDerivedObligation(ref cause) => {
                tcx.lift(cause).map(super::ImplDerivedObligation)
            }
            super::DerivedObligation(ref cause) => tcx.lift(cause).map(super::DerivedObligation),
            super::CompareImplConstObligation => Some(super::CompareImplConstObligation),
            super::CompareImplMethodObligation {
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            } => Some(super::CompareImplMethodObligation {
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            }),
            super::CompareImplTypeObligation { item_name, impl_item_def_id, trait_item_def_id } => {
                Some(super::CompareImplTypeObligation {
                    item_name,
                    impl_item_def_id,
                    trait_item_def_id,
                })
            }
            super::ExprAssignable => Some(super::ExprAssignable),
            super::MatchExpressionArm(box super::MatchExpressionArmCause {
                arm_span,
                source,
                ref prior_arms,
                last_ty,
                scrut_hir_id,
            }) => tcx.lift(&last_ty).map(|last_ty| {
                super::MatchExpressionArm(box super::MatchExpressionArmCause {
                    arm_span,
                    source,
                    prior_arms: prior_arms.clone(),
                    last_ty,
                    scrut_hir_id,
                })
            }),
            super::Pattern { span, root_ty, origin_expr } => {
                tcx.lift(&root_ty).map(|root_ty| super::Pattern { span, root_ty, origin_expr })
            }
            super::IfExpression(box super::IfExpressionCause { then, outer, semicolon }) => {
                Some(super::IfExpression(box super::IfExpressionCause { then, outer, semicolon }))
            }
            super::IfExpressionWithNoElse => Some(super::IfExpressionWithNoElse),
            super::MainFunctionType => Some(super::MainFunctionType),
            super::StartFunctionType => Some(super::StartFunctionType),
            super::IntrinsicType => Some(super::IntrinsicType),
            super::MethodReceiver => Some(super::MethodReceiver),
            super::BlockTailExpression(id) => Some(super::BlockTailExpression(id)),
            super::TrivialBound => Some(super::TrivialBound),
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::DerivedObligationCause<'a> {
    type Lifted = traits::DerivedObligationCause<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.parent_trait_ref).and_then(|trait_ref| {
            tcx.lift(&*self.parent_code).map(|code| traits::DerivedObligationCause {
                parent_trait_ref: trait_ref,
                parent_code: Rc::new(code),
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::ObligationCause<'a> {
    type Lifted = traits::ObligationCause<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.code).map(|code| traits::ObligationCause {
            span: self.span,
            body_id: self.body_id,
            code,
        })
    }
}

// For codegen only.
impl<'a, 'tcx> Lift<'tcx> for traits::Vtable<'a, ()> {
    type Lifted = traits::Vtable<'tcx, ()>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self.clone() {
            traits::VtableImpl(traits::VtableImplData { impl_def_id, substs, nested }) => {
                tcx.lift(&substs).map(|substs| {
                    traits::VtableImpl(traits::VtableImplData { impl_def_id, substs, nested })
                })
            }
            traits::VtableAutoImpl(t) => Some(traits::VtableAutoImpl(t)),
            traits::VtableGenerator(traits::VtableGeneratorData {
                generator_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::VtableGenerator(traits::VtableGeneratorData {
                    generator_def_id,
                    substs,
                    nested,
                })
            }),
            traits::VtableClosure(traits::VtableClosureData { closure_def_id, substs, nested }) => {
                tcx.lift(&substs).map(|substs| {
                    traits::VtableClosure(traits::VtableClosureData {
                        closure_def_id,
                        substs,
                        nested,
                    })
                })
            }
            traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested }) => {
                tcx.lift(&fn_ty).map(|fn_ty| {
                    traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested })
                })
            }
            traits::VtableDiscriminantKind(traits::VtableDiscriminantKindData) => {
                Some(traits::VtableDiscriminantKind(traits::VtableDiscriminantKindData))
            }
            traits::VtableParam(n) => Some(traits::VtableParam(n)),
            traits::VtableBuiltin(n) => Some(traits::VtableBuiltin(n)),
            traits::VtableObject(traits::VtableObjectData {
                upcast_trait_ref,
                vtable_base,
                nested,
            }) => tcx.lift(&upcast_trait_ref).map(|trait_ref| {
                traits::VtableObject(traits::VtableObjectData {
                    upcast_trait_ref: trait_ref,
                    vtable_base,
                    nested,
                })
            }),
            traits::VtableTraitAlias(traits::VtableTraitAliasData {
                alias_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::VtableTraitAlias(traits::VtableTraitAliasData {
                    alias_def_id,
                    substs,
                    nested,
                })
            }),
        }
    }
}
