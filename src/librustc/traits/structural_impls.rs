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
use ty::{self, Lift, TyCtxt};
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

impl<'tcx, O: fmt::Debug> fmt::Debug for traits::Obligation<'tcx, O> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if ty::tls::with(|tcx| tcx.sess.verbose()) {
            write!(f, "Obligation(predicate={:?},cause={:?},depth={})",
                   self.predicate,
                   self.cause,
                   self.recursion_depth)
        } else {
            write!(f, "Obligation(predicate={:?},depth={})",
                   self.predicate,
                   self.recursion_depth)
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::Vtable<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::VtableImpl(ref v) =>
                write!(f, "{:?}", v),

            super::VtableAutoImpl(ref t) =>
                write!(f, "{:?}", t),

            super::VtableClosure(ref d) =>
                write!(f, "{:?}", d),

            super::VtableGenerator(ref d) =>
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

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableGeneratorData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableGenerator(closure_def_id={:?}, substs={:?}, nested={:?})",
               self.closure_def_id,
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

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableAutoImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableAutoImplData(trait_def_id={:?}, nested={:?})",
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
            super::CodeSubtypeError(ref a, ref b) =>
                write!(f, "CodeSubtypeError({:?}, {:?})", a, b),
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
            super::ConstEvalFailure(ref err) => {
                tcx.lift(err).map(super::ConstEvalFailure)
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::ObligationCauseCode<'a> {
    type Lifted = traits::ObligationCauseCode<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            super::ReturnNoExpression => Some(super::ReturnNoExpression),
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
            super::TupleInitializerSized => Some(super::TupleInitializerSized),
            super::StructInitializerSized => Some(super::StructInitializerSized),
            super::VariableType(id) => Some(super::VariableType(id)),
            super::ReturnType(id) => Some(super::ReturnType(id)),
            super::SizedReturnType => Some(super::SizedReturnType),
            super::SizedYieldType => Some(super::SizedYieldType),
            super::RepeatVec => Some(super::RepeatVec),
            super::FieldSized(item) => Some(super::FieldSized(item)),
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
                                                 trait_item_def_id } => {
                Some(super::CompareImplMethodObligation {
                    item_name,
                    impl_item_def_id,
                    trait_item_def_id,
                })
            }
            super::ExprAssignable => Some(super::ExprAssignable),
            super::MatchExpressionArm { arm_span, source } => {
                Some(super::MatchExpressionArm { arm_span,
                                                 source: source })
            }
            super::IfExpression => Some(super::IfExpression),
            super::IfExpressionWithNoElse => Some(super::IfExpressionWithNoElse),
            super::MainFunctionType => Some(super::MainFunctionType),
            super::StartFunctionType => Some(super::StartFunctionType),
            super::IntrinsicType => Some(super::IntrinsicType),
            super::MethodReceiver => Some(super::MethodReceiver),
            super::BlockTailExpression(id) => Some(super::BlockTailExpression(id)),
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
                code,
            }
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
                        impl_def_id,
                        substs,
                        nested,
                    })
                })
            }
            traits::VtableAutoImpl(t) => Some(traits::VtableAutoImpl(t)),
            traits::VtableGenerator(traits::VtableGeneratorData {
                closure_def_id,
                substs,
                nested
            }) => {
                tcx.lift(&substs).map(|substs| {
                    traits::VtableGenerator(traits::VtableGeneratorData {
                        closure_def_id: closure_def_id,
                        substs: substs,
                        nested: nested
                    })
                })
            }
            traits::VtableClosure(traits::VtableClosureData {
                closure_def_id,
                substs,
                nested
            }) => {
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
                    traits::VtableFnPointer(traits::VtableFnPointerData {
                        fn_ty,
                        nested,
                    })
                })
            }
            traits::VtableParam(n) => Some(traits::VtableParam(n)),
            traits::VtableBuiltin(n) => Some(traits::VtableBuiltin(n)),
            traits::VtableObject(traits::VtableObjectData {
                upcast_trait_ref,
                vtable_base,
                nested
            }) => {
                tcx.lift(&upcast_trait_ref).map(|trait_ref| {
                    traits::VtableObject(traits::VtableObjectData {
                        upcast_trait_ref: trait_ref,
                        vtable_base,
                        nested,
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
            param_env: self.param_env.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicate.visit_with(visitor)
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableImplData<'tcx, N> {
        impl_def_id, substs, nested
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableGeneratorData<'tcx, N> {
        closure_def_id, substs, nested
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableClosureData<'tcx, N> {
        closure_def_id, substs, nested
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableAutoImplData<N> {
        trait_def_id, nested
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableBuiltinData<N> {
        nested
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableObjectData<'tcx, N> {
        upcast_trait_ref, vtable_base, nested
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableFnPointerData<'tcx, N> {
        fn_ty,
        nested
    } where N: TypeFoldable<'tcx>
}

EnumTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::Vtable<'tcx, N> {
        (traits::VtableImpl)(a),
        (traits::VtableAutoImpl)(a),
        (traits::VtableGenerator)(a),
        (traits::VtableClosure)(a),
        (traits::VtableFnPointer)(a),
        (traits::VtableParam)(a),
        (traits::VtableBuiltin)(a),
        (traits::VtableObject)(a),
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for Normalized<'tcx, T> {
        value,
        obligations
    } where T: TypeFoldable<'tcx>
}

impl<'tcx> fmt::Display for traits::WhereClauseAtom<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::WhereClauseAtom::*;
        match *self {
            Implemented(ref trait_ref) => write!(fmt, "Implemented({})", trait_ref),
            ProjectionEq(ref projection) => write!(fmt, "ProjectionEq({})", projection),
        }
    }
}

impl<'tcx> fmt::Display for traits::DomainGoal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::DomainGoal::*;
        use traits::WhereClauseAtom::*;
        match *self {
            Holds(wc) => write!(fmt, "{}", wc),
            WellFormed(Implemented(ref trait_ref)) => write!(fmt, "WellFormed({})", trait_ref),
            WellFormed(ProjectionEq(ref projection)) => write!(fmt, "WellFormed({})", projection),
            FromEnv(Implemented(ref trait_ref)) => write!(fmt, "FromEnv({})", trait_ref),
            FromEnv(ProjectionEq(ref projection)) => write!(fmt, "FromEnv({})", projection),
            WellFormedTy(ref ty) => write!(fmt, "WellFormed({})", ty),
            FromEnvTy(ref ty) => write!(fmt, "FromEnv({})", ty),
            RegionOutlives(ref predicate) => write!(fmt, "RegionOutlives({})", predicate),
            TypeOutlives(ref predicate) => write!(fmt, "TypeOutlives({})", predicate),
        }
    }
}

impl fmt::Display for traits::QuantifierKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::QuantifierKind::*;
        match *self {
            Universal => write!(fmt, "forall"),
            Existential => write!(fmt, "exists"),
        }
    }
}

impl<'tcx> fmt::Display for traits::LeafGoal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::LeafGoal::*;
        match *self {
            DomainGoal(ref domain_goal) => write!(fmt, "{}", domain_goal),
        }
    }
}

impl<'tcx> fmt::Display for traits::Goal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::Goal::*;
        match *self {
            Implies(ref hypotheses, ref goal) => {
                write!(fmt, "if (")?;
                for (index, hyp) in hypotheses.iter().enumerate() {
                    if index > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{}", hyp)?;
                }
                write!(fmt, ") {{ {} }}", goal)
            }
            And(ref goal1, ref goal2) => write!(fmt, "({}, {})", goal1, goal2),
            Not(ref goal) => write!(fmt, "not {{ {} }}", goal),
            Leaf(ref goal) => write!(fmt, "{}", goal),
            Quantified(qkind, ref goal) => {
                // FIXME: appropriate binder names
                write!(fmt, "{}<> {{ {} }}", qkind, goal.skip_binder())
            }
        }
    }
}

impl<'tcx> fmt::Display for traits::ProgramClause<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.consequence)?;
        if self.conditions.is_empty() {
            write!(fmt, ".")?;
        } else {
            write!(fmt, " :- ")?;
            for (index, condition) in self.conditions.iter().enumerate() {
                if index > 0 {
                    write!(fmt, ", ")?;
                }
                write!(fmt, "{}", condition)?;
            }
        }
        Ok(())
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::WhereClauseAtom<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        use traits::WhereClauseAtom::*;
        match *self {
            Implemented(ref trait_ref) => Implemented(trait_ref.fold_with(folder)),
            ProjectionEq(ref projection) => ProjectionEq(projection.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        use traits::WhereClauseAtom::*;
        match *self {
            Implemented(ref trait_ref) => trait_ref.visit_with(visitor),
            ProjectionEq(ref projection) => projection.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::DomainGoal<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        use traits::DomainGoal::*;
        match *self {
            Holds(ref wc) => Holds(wc.fold_with(folder)),
            WellFormed(ref wc) => WellFormed(wc.fold_with(folder)),
            FromEnv(ref wc) => FromEnv(wc.fold_with(folder)),
            WellFormedTy(ref ty) => WellFormedTy(ty.fold_with(folder)),
            FromEnvTy(ref ty) => FromEnvTy(ty.fold_with(folder)),
            RegionOutlives(ref predicate) => RegionOutlives(predicate.fold_with(folder)),
            TypeOutlives(ref predicate) => TypeOutlives(predicate.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        use traits::DomainGoal::*;
        match *self {
            Holds(ref wc) |
            WellFormed(ref wc) |
            FromEnv(ref wc) => wc.visit_with(visitor),
            WellFormedTy(ref ty) |
            FromEnvTy(ref ty) => ty.visit_with(visitor),
            RegionOutlives(ref predicate) => predicate.visit_with(visitor),
            TypeOutlives(ref predicate) => predicate.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::LeafGoal<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        use traits::LeafGoal::*;
        match *self {
            DomainGoal(ref domain_goal) => DomainGoal(domain_goal.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        use traits::LeafGoal::*;
        match *self {
            DomainGoal(ref domain_goal) => domain_goal.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::Goal<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        use traits::Goal::*;
        match *self {
            Implies(ref hypotheses, ref goal) => {
                Implies(
                    hypotheses.iter().map(|hyp| hyp.fold_with(folder)).collect(),
                    goal.fold_with(folder)
                )
            },
            And(ref goal1, ref goal2) => And(goal1.fold_with(folder), goal2.fold_with(folder)),
            Not(ref goal) => Not(goal.fold_with(folder)),
            Leaf(ref leaf_goal) => Leaf(leaf_goal.fold_with(folder)),
            Quantified(qkind, ref goal) => Quantified(qkind, goal.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        use traits::Goal::*;
        match *self {
            Implies(ref hypotheses, ref goal) => {
                hypotheses.iter().any(|hyp| hyp.visit_with(visitor)) || goal.visit_with(visitor)
            }
            And(ref goal1, ref goal2) => goal1.visit_with(visitor) || goal2.visit_with(visitor),
            Not(ref goal) => goal.visit_with(visitor),
            Leaf(ref leaf_goal) => leaf_goal.visit_with(visitor),
            Quantified(_, ref goal) => goal.visit_with(visitor),
        }
    }
}
