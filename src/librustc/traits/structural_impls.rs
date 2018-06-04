// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use chalk_engine;
use rustc_data_structures::accumulate_vec::AccumulateVec;
use traits;
use traits::project::Normalized;
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use ty::{self, Lift, TyCtxt};

use std::fmt;
use std::rc::Rc;

// structural impls for the structs in traits

impl<'tcx, T: fmt::Debug> fmt::Debug for Normalized<'tcx, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Normalized({:?},{:?})", self.value, self.obligations)
    }
}

impl<'tcx, O: fmt::Debug> fmt::Debug for traits::Obligation<'tcx, O> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if ty::tls::with(|tcx| tcx.sess.verbose()) {
            write!(
                f,
                "Obligation(predicate={:?},cause={:?},depth={})",
                self.predicate, self.cause, self.recursion_depth
            )
        } else {
            write!(
                f,
                "Obligation(predicate={:?},depth={})",
                self.predicate, self.recursion_depth
            )
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::Vtable<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::VtableImpl(ref v) => write!(f, "{:?}", v),

            super::VtableAutoImpl(ref t) => write!(f, "{:?}", t),

            super::VtableClosure(ref d) => write!(f, "{:?}", d),

            super::VtableGenerator(ref d) => write!(f, "{:?}", d),

            super::VtableFnPointer(ref d) => write!(f, "VtableFnPointer({:?})", d),

            super::VtableObject(ref d) => write!(f, "{:?}", d),

            super::VtableParam(ref n) => write!(f, "VtableParam({:?})", n),

            super::VtableBuiltin(ref d) => write!(f, "{:?}", d),
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableImplData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VtableImpl(impl_def_id={:?}, substs={:?}, nested={:?})",
            self.impl_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableGeneratorData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VtableGenerator(generator_def_id={:?}, substs={:?}, nested={:?})",
            self.generator_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableClosureData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VtableClosure(closure_def_id={:?}, substs={:?}, nested={:?})",
            self.closure_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableBuiltinData<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableBuiltin(nested={:?})", self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableAutoImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VtableAutoImplData(trait_def_id={:?}, nested={:?})",
            self.trait_def_id, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableObjectData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VtableObject(upcast={:?}, vtable_base={}, nested={:?})",
            self.upcast_trait_ref, self.vtable_base, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::VtableFnPointerData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VtableFnPointer(fn_ty={:?}, nested={:?})",
            self.fn_ty, self.nested
        )
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FulfillmentError({:?},{:?})", self.obligation, self.code)
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::CodeSelectionError(ref e) => write!(f, "{:?}", e),
            super::CodeProjectionError(ref e) => write!(f, "{:?}", e),
            super::CodeSubtypeError(ref a, ref b) => {
                write!(f, "CodeSubtypeError({:?}, {:?})", a, b)
            }
            super::CodeAmbiguity => write!(f, "Ambiguity"),
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
                    tcx.lift(err)
                        .map(|err| super::OutputTypeParameterMismatch(a, b, err))
                })
            }
            super::TraitNotObjectSafe(def_id) => Some(super::TraitNotObjectSafe(def_id)),
            super::ConstEvalFailure(ref err) => tcx.lift(err).map(super::ConstEvalFailure),
            super::Overflow => bug!(), // FIXME: ape ConstEvalFailure?
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
            super::ObjectTypeBound(ty, r) => tcx.lift(&ty).and_then(|ty| {
                tcx.lift(&r)
                    .and_then(|r| Some(super::ObjectTypeBound(ty, r)))
            }),
            super::ObjectCastObligation(ty) => tcx.lift(&ty).map(super::ObjectCastObligation),
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
            super::CompareImplMethodObligation {
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            } => Some(super::CompareImplMethodObligation {
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            }),
            super::ExprAssignable => Some(super::ExprAssignable),
            super::MatchExpressionArm { arm_span, source } => Some(super::MatchExpressionArm {
                arm_span,
                source: source,
            }),
            super::IfExpression => Some(super::IfExpression),
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
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.parent_trait_ref).and_then(|trait_ref| {
            tcx.lift(&*self.parent_code)
                .map(|code| traits::DerivedObligationCause {
                    parent_trait_ref: trait_ref,
                    parent_code: Rc::new(code),
                })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::ObligationCause<'a> {
    type Lifted = traits::ObligationCause<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
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
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match self.clone() {
            traits::VtableImpl(traits::VtableImplData {
                impl_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::VtableImpl(traits::VtableImplData {
                    impl_def_id,
                    substs,
                    nested,
                })
            }),
            traits::VtableAutoImpl(t) => Some(traits::VtableAutoImpl(t)),
            traits::VtableGenerator(traits::VtableGeneratorData {
                generator_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::VtableGenerator(traits::VtableGeneratorData {
                    generator_def_id: generator_def_id,
                    substs: substs,
                    nested: nested,
                })
            }),
            traits::VtableClosure(traits::VtableClosureData {
                closure_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs| {
                traits::VtableClosure(traits::VtableClosureData {
                    closure_def_id,
                    substs,
                    nested,
                })
            }),
            traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested }) => {
                tcx.lift(&fn_ty).map(|fn_ty| {
                    traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested })
                })
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
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.

impl<'tcx, O: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Obligation<'tcx, O> {
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
        generator_def_id, substs, nested
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

impl<'tcx> fmt::Display for traits::WhereClause<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::WhereClause::*;

        match self {
            Implemented(trait_ref) => write!(fmt, "Implemented({})", trait_ref),
            ProjectionEq(projection) => write!(fmt, "ProjectionEq({})", projection),
            RegionOutlives(predicate) => write!(fmt, "RegionOutlives({})", predicate),
            TypeOutlives(predicate) => write!(fmt, "TypeOutlives({})", predicate),
        }
    }
}

impl<'tcx> fmt::Display for traits::WellFormed<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::WellFormed::*;

        match self {
            Trait(trait_ref) => write!(fmt, "WellFormed({})", trait_ref),
            Ty(ty) => write!(fmt, "WellFormed({})", ty),
        }
    }
}

impl<'tcx> fmt::Display for traits::FromEnv<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::FromEnv::*;

        match self {
            Trait(trait_ref) => write!(fmt, "FromEnv({})", trait_ref),
            Ty(ty) => write!(fmt, "FromEnv({})", ty),
        }
    }
}

impl<'tcx> fmt::Display for traits::DomainGoal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::DomainGoal::*;

        match self {
            Holds(wc) => write!(fmt, "{}", wc),
            WellFormed(wf) => write!(fmt, "{}", wf),
            FromEnv(from_env) => write!(fmt, "{}", from_env),
            Normalize(projection) => write!(fmt, "Normalize({})", projection),
        }
    }
}

impl fmt::Display for traits::QuantifierKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::QuantifierKind::*;

        match self {
            Universal => write!(fmt, "forall"),
            Existential => write!(fmt, "exists"),
        }
    }
}

impl<'tcx> fmt::Display for traits::Goal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::Goal::*;

        match self {
            Implies(hypotheses, goal) => {
                write!(fmt, "if (")?;
                for (index, hyp) in hypotheses.iter().enumerate() {
                    if index > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{}", hyp)?;
                }
                write!(fmt, ") {{ {} }}", goal)
            }
            And(goal1, goal2) => write!(fmt, "({} && {})", goal1, goal2),
            Not(goal) => write!(fmt, "not {{ {} }}", goal),
            DomainGoal(goal) => write!(fmt, "{}", goal),
            Quantified(qkind, goal) => {
                // FIXME: appropriate binder names
                write!(fmt, "{}<> {{ {} }}", qkind, goal.skip_binder())
            }
            CannotProve => write!(fmt, "CannotProve"),
        }
    }
}

impl<'tcx> fmt::Display for traits::ProgramClause<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let traits::ProgramClause { goal, hypotheses } = self;
        write!(fmt, "{}", goal)?;
        if !hypotheses.is_empty() {
            write!(fmt, " :- ")?;
            for (index, condition) in hypotheses.iter().enumerate() {
                if index > 0 {
                    write!(fmt, ", ")?;
                }
                write!(fmt, "{}", condition)?;
            }
        }
        write!(fmt, ".")
    }
}

impl<'tcx> fmt::Display for traits::Clause<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use traits::Clause::*;

        match self {
            Implies(clause) => write!(fmt, "{}", clause),
            ForAll(clause) => {
                // FIXME: appropriate binder names
                write!(fmt, "forall<> {{ {} }}", clause.skip_binder())
            }
        }
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::WhereClause<'tcx> {
        (traits::WhereClause::Implemented)(trait_ref),
        (traits::WhereClause::ProjectionEq)(projection),
        (traits::WhereClause::TypeOutlives)(ty_outlives),
        (traits::WhereClause::RegionOutlives)(region_outlives),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::WhereClause<'a> {
        type Lifted = traits::WhereClause<'tcx>;
        (traits::WhereClause::Implemented)(trait_ref),
        (traits::WhereClause::ProjectionEq)(projection),
        (traits::WhereClause::TypeOutlives)(ty_outlives),
        (traits::WhereClause::RegionOutlives)(region_outlives),
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::WellFormed<'tcx> {
        (traits::WellFormed::Trait)(trait_ref),
        (traits::WellFormed::Ty)(ty),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::WellFormed<'a> {
        type Lifted = traits::WellFormed<'tcx>;
        (traits::WellFormed::Trait)(trait_ref),
        (traits::WellFormed::Ty)(ty),
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::FromEnv<'tcx> {
        (traits::FromEnv::Trait)(trait_ref),
        (traits::FromEnv::Ty)(ty),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::FromEnv<'a> {
        type Lifted = traits::FromEnv<'tcx>;
        (traits::FromEnv::Trait)(trait_ref),
        (traits::FromEnv::Ty)(ty),
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::DomainGoal<'tcx> {
        (traits::DomainGoal::Holds)(wc),
        (traits::DomainGoal::WellFormed)(wf),
        (traits::DomainGoal::FromEnv)(from_env),
        (traits::DomainGoal::Normalize)(projection),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::DomainGoal<'a> {
        type Lifted = traits::DomainGoal<'tcx>;
        (traits::DomainGoal::Holds)(wc),
        (traits::DomainGoal::WellFormed)(wf),
        (traits::DomainGoal::FromEnv)(from_env),
        (traits::DomainGoal::Normalize)(projection),
    }
}

CloneTypeFoldableAndLiftImpls! {
    traits::QuantifierKind,
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::Goal<'tcx> {
        (traits::Goal::Implies)(hypotheses, goal),
        (traits::Goal::And)(goal1, goal2),
        (traits::Goal::Not)(goal),
        (traits::Goal::DomainGoal)(domain_goal),
        (traits::Goal::Quantified)(qkind, goal),
        (traits::Goal::CannotProve),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::Goal<'a> {
        type Lifted = traits::Goal<'tcx>;
        (traits::Goal::Implies)(hypotheses, goal),
        (traits::Goal::And)(goal1, goal2),
        (traits::Goal::Not)(goal),
        (traits::Goal::DomainGoal)(domain_goal),
        (traits::Goal::Quantified)(kind, goal),
        (traits::Goal::CannotProve),
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Slice<traits::Goal<'tcx>> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter()
            .map(|t| t.fold_with(folder))
            .collect::<AccumulateVec<[_; 8]>>();
        folder.tcx().intern_goals(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx traits::Goal<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let v = (**self).fold_with(folder);
        folder.tcx().mk_goal(v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::ProgramClause<'tcx> {
        goal,
        hypotheses
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::Clause<'tcx> {
        (traits::Clause::Implies)(clause),
        (traits::Clause::ForAll)(clause),
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Slice<traits::Clause<'tcx>> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter()
            .map(|t| t.fold_with(folder))
            .collect::<AccumulateVec<[_; 8]>>();
        folder.tcx().intern_clauses(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, C> TypeFoldable<'tcx> for chalk_engine::ExClause<C>
where
    C: traits::ExClauseFold<'tcx>,
    C::Substitution: Clone,
    C::RegionConstraint: Clone,
{
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        <C as traits::ExClauseFold>::fold_ex_clause_with(
            self,
            folder,
        )
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        <C as traits::ExClauseFold>::visit_ex_clause_with(
            self,
            visitor,
        )
    }
}

impl<'tcx, C> Lift<'tcx> for chalk_engine::ExClause<C>
where
    C: chalk_engine::context::Context + Clone,
    C: traits::ExClauseLift<'tcx>,
{
    type Lifted = C::LiftedExClause;

    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        <C as traits::ExClauseLift>::lift_ex_clause_to_tcx(self, tcx)
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx, C> TypeFoldable<'tcx> for chalk_engine::DelayedLiteral<C> {
        (chalk_engine::DelayedLiteral::CannotProve)(a),
        (chalk_engine::DelayedLiteral::Negative)(a),
        (chalk_engine::DelayedLiteral::Positive)(a, b),
    } where
        C: chalk_engine::context::Context + Clone,
        C::CanonicalConstrainedSubst: TypeFoldable<'tcx>,
}

EnumTypeFoldableImpl! {
    impl<'tcx, C> TypeFoldable<'tcx> for chalk_engine::Literal<C> {
        (chalk_engine::Literal::Negative)(a),
        (chalk_engine::Literal::Positive)(a),
    } where
        C: chalk_engine::context::Context + Clone,
        C::GoalInEnvironment: Clone + TypeFoldable<'tcx>,
}

CloneTypeFoldableAndLiftImpls! {
    chalk_engine::TableIndex,
}
