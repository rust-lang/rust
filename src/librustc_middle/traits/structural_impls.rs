use crate::traits;
use crate::ty::{Lift, TyCtxt};

use std::fmt;
use std::rc::Rc;

// Structural impls for the structs in `traits`.

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSource<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::ImplSourceUserDefined(ref v) => write!(f, "{:?}", v),

            super::ImplSourceAutoImpl(ref t) => write!(f, "{:?}", t),

            super::ImplSourceClosure(ref d) => write!(f, "{:?}", d),

            super::ImplSourceGenerator(ref d) => write!(f, "{:?}", d),

            super::ImplSourceFnPointer(ref d) => write!(f, "ImplSourceFnPointer({:?})", d),

            super::ImplSourceDiscriminantKind(ref d) => write!(f, "{:?}", d),

            super::ImplSourceObject(ref d) => write!(f, "{:?}", d),

            super::ImplSourceParam(ref n) => write!(f, "ImplSourceParam({:?})", n),

            super::ImplSourceBuiltin(ref d) => write!(f, "{:?}", d),

            super::ImplSourceTraitAlias(ref d) => write!(f, "{:?}", d),
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceUserDefinedData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceUserDefinedData(impl_def_id={:?}, substs={:?}, nested={:?})",
            self.impl_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceGeneratorData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceGeneratorData(generator_def_id={:?}, substs={:?}, nested={:?})",
            self.generator_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceClosureData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceClosureData(closure_def_id={:?}, substs={:?}, nested={:?})",
            self.closure_def_id, self.substs, self.nested
        )
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::ImplSourceBuiltinData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImplSourceBuiltinData(nested={:?})", self.nested)
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::ImplSourceAutoImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceAutoImplData(trait_def_id={:?}, nested={:?})",
            self.trait_def_id, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceObjectData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceObjectData(upcast={:?}, vtable_base={}, nested={:?})",
            self.upcast_trait_ref, self.vtable_base, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceFnPointerData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImplSourceFnPointerData(fn_ty={:?}, nested={:?})", self.fn_ty, self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceTraitAliasData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceTraitAlias(alias_def_id={:?}, substs={:?}, nested={:?})",
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
            super::SizedArgumentType(sp) => Some(super::SizedArgumentType(sp)),
            super::SizedReturnType => Some(super::SizedReturnType),
            super::SizedYieldType => Some(super::SizedYieldType),
            super::InlineAsmSized => Some(super::InlineAsmSized),
            super::RepeatVec(suggest_flag) => Some(super::RepeatVec(suggest_flag)),
            super::FieldSized { adt_kind, span, last } => {
                Some(super::FieldSized { adt_kind, span, last })
            }
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
        tcx.lift(&self.code).map(|code| traits::ObligationCause::new(self.span, self.body_id, code))
    }
}

// For codegen only.
impl<'a, 'tcx> Lift<'tcx> for traits::ImplSource<'a, ()> {
    type Lifted = traits::ImplSource<'tcx, ()>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self.clone() {
            traits::ImplSourceUserDefined(traits::ImplSourceUserDefinedData {
                impl_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::ImplSourceUserDefined(traits::ImplSourceUserDefinedData {
                    impl_def_id,
                    substs,
                    nested,
                })
            }),
            traits::ImplSourceAutoImpl(t) => Some(traits::ImplSourceAutoImpl(t)),
            traits::ImplSourceGenerator(traits::ImplSourceGeneratorData {
                generator_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::ImplSourceGenerator(traits::ImplSourceGeneratorData {
                    generator_def_id,
                    substs,
                    nested,
                })
            }),
            traits::ImplSourceClosure(traits::ImplSourceClosureData {
                closure_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::ImplSourceClosure(traits::ImplSourceClosureData {
                    closure_def_id,
                    substs,
                    nested,
                })
            }),
            traits::ImplSourceFnPointer(traits::ImplSourceFnPointerData { fn_ty, nested }) => {
                tcx.lift(&fn_ty).map(|fn_ty| {
                    traits::ImplSourceFnPointer(traits::ImplSourceFnPointerData { fn_ty, nested })
                })
            }
            traits::ImplSourceDiscriminantKind(traits::ImplSourceDiscriminantKindData) => {
                Some(traits::ImplSourceDiscriminantKind(traits::ImplSourceDiscriminantKindData))
            }
            traits::ImplSourceParam(n) => Some(traits::ImplSourceParam(n)),
            traits::ImplSourceBuiltin(n) => Some(traits::ImplSourceBuiltin(n)),
            traits::ImplSourceObject(traits::ImplSourceObjectData {
                upcast_trait_ref,
                vtable_base,
                nested,
            }) => tcx.lift(&upcast_trait_ref).map(|trait_ref| {
                traits::ImplSourceObject(traits::ImplSourceObjectData {
                    upcast_trait_ref: trait_ref,
                    vtable_base,
                    nested,
                })
            }),
            traits::ImplSourceTraitAlias(traits::ImplSourceTraitAliasData {
                alias_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::ImplSourceTraitAlias(traits::ImplSourceTraitAliasData {
                    alias_def_id,
                    substs,
                    nested,
                })
            }),
        }
    }
}
