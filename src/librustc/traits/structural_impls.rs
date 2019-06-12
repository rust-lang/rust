use chalk_engine;
use smallvec::SmallVec;
use crate::traits;
use crate::traits::project::Normalized;
use crate::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use crate::ty::{self, Lift, Ty, TyCtxt};
use syntax::symbol::InternedString;

use std::fmt;
use std::rc::Rc;
use std::collections::{BTreeSet, BTreeMap};

// Structural impls for the structs in `traits`.

impl<'tcx, T: fmt::Debug> fmt::Debug for Normalized<'tcx, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Normalized({:?}, {:?})", self.value, self.obligations)
    }
}

impl<'tcx, O: fmt::Debug> fmt::Debug for traits::Obligation<'tcx, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if ty::tls::with(|tcx| tcx.sess.verbose()) {
            write!(
                f,
                "Obligation(predicate={:?}, cause={:?}, param_env={:?}, depth={})",
                self.predicate, self.cause, self.param_env, self.recursion_depth
            )
        } else {
            write!(
                f,
                "Obligation(predicate={:?}, depth={})",
                self.predicate, self.recursion_depth
            )
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::Vtable<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::VtableImpl(ref v) => write!(f, "{:?}", v),

            super::VtableAutoImpl(ref t) => write!(f, "{:?}", t),

            super::VtableClosure(ref d) => write!(f, "{:?}", d),

            super::VtableGenerator(ref d) => write!(f, "{:?}", d),

            super::VtableFnPointer(ref d) => write!(f, "VtableFnPointer({:?})", d),

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
        write!(
            f,
            "VtableFnPointerData(fn_ty={:?}, nested={:?})",
            self.fn_ty, self.nested
        )
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

impl<'tcx> fmt::Debug for traits::FulfillmentError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FulfillmentError({:?},{:?})", self.obligation, self.code)
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MismatchedProjectionTypes({:?})", self.err)
    }
}

impl<'tcx> fmt::Display for traits::WhereClause<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::WhereClause::*;

        // Bypass `ty::print` because it does not print out anonymous regions.
        // FIXME(eddyb) implement a custom `PrettyPrinter`, or move this to `ty::print`.
        fn write_region_name<'tcx>(
            r: ty::Region<'tcx>,
            fmt: &mut fmt::Formatter<'_>
        ) -> fmt::Result {
            match r {
                ty::ReLateBound(index, br) => match br {
                    ty::BoundRegion::BrNamed(_, name) => write!(fmt, "{}", name),
                    ty::BoundRegion::BrAnon(var) => {
                        if *index == ty::INNERMOST {
                            write!(fmt, "'^{}", var)
                        } else {
                            write!(fmt, "'^{}_{}", index.index(), var)
                        }
                    }
                    _ => write!(fmt, "'_"),
                }

                _ => write!(fmt, "{}", r),
            }
        }

        match self {
            Implemented(trait_ref) => write!(fmt, "Implemented({})", trait_ref),
            ProjectionEq(projection) => write!(fmt, "ProjectionEq({})", projection),
            RegionOutlives(predicate) => {
                write!(fmt, "RegionOutlives({}: ", predicate.0)?;
                write_region_name(predicate.1, fmt)?;
                write!(fmt, ")")
            }
            TypeOutlives(predicate) => {
                write!(fmt, "TypeOutlives({}: ", predicate.0)?;
                write_region_name(predicate.1, fmt)?;
                write!(fmt, ")")
            }
        }
    }
}

impl<'tcx> fmt::Display for traits::WellFormed<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::WellFormed::*;

        match self {
            Trait(trait_ref) => write!(fmt, "WellFormed({})", trait_ref),
            Ty(ty) => write!(fmt, "WellFormed({})", ty),
        }
    }
}

impl<'tcx> fmt::Display for traits::FromEnv<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::FromEnv::*;

        match self {
            Trait(trait_ref) => write!(fmt, "FromEnv({})", trait_ref),
            Ty(ty) => write!(fmt, "FromEnv({})", ty),
        }
    }
}

impl<'tcx> fmt::Display for traits::DomainGoal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::DomainGoal::*;

        match self {
            Holds(wc) => write!(fmt, "{}", wc),
            WellFormed(wf) => write!(fmt, "{}", wf),
            FromEnv(from_env) => write!(fmt, "{}", from_env),
            Normalize(projection) => write!(
                fmt,
                "Normalize({} -> {})",
                projection.projection_ty,
                projection.ty
            ),
        }
    }
}

impl fmt::Display for traits::QuantifierKind {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::QuantifierKind::*;

        match self {
            Universal => write!(fmt, "forall"),
            Existential => write!(fmt, "exists"),
        }
    }
}

/// Collect names for regions / types bound by a quantified goal / clause.
/// This collector does not try to do anything clever like in `ty::print`, it's just used
/// for debug output in tests anyway.
struct BoundNamesCollector {
    // Just sort by name because `BoundRegion::BrNamed` does not have a `BoundVar` index anyway.
    regions: BTreeSet<InternedString>,

    // Sort by `BoundVar` index, so usually this should be equivalent to the order given
    // by the list of type parameters.
    types: BTreeMap<u32, InternedString>,

    binder_index: ty::DebruijnIndex,
}

impl BoundNamesCollector {
    fn new() -> Self {
        BoundNamesCollector {
            regions: BTreeSet::new(),
            types: BTreeMap::new(),
            binder_index: ty::INNERMOST,
        }
    }

    fn is_empty(&self) -> bool {
        self.regions.is_empty() && self.types.is_empty()
    }

    fn write_names(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut start = true;
        for r in &self.regions {
            if !start {
                write!(fmt, ", ")?;
            }
            start = false;
            write!(fmt, "{}", r)?;
        }
        for (_, t) in &self.types {
            if !start {
                write!(fmt, ", ")?;
            }
            start = false;
            write!(fmt, "{}", t)?;
        }
        Ok(())
    }
}

impl<'tcx> TypeVisitor<'tcx> for BoundNamesCollector {
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> bool {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        match t.sty {
            ty::Bound(debruijn, bound_ty) if debruijn == self.binder_index => {
                self.types.insert(
                    bound_ty.var.as_u32(),
                    match bound_ty.kind {
                        ty::BoundTyKind::Param(name) => name,
                        ty::BoundTyKind::Anon =>
                            InternedString::intern(&format!("^{}", bound_ty.var.as_u32()),
                        ),
                    }
                );
            }

            _ => (),
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        match r {
            ty::ReLateBound(index, br) if *index == self.binder_index => {
                match br {
                    ty::BoundRegion::BrNamed(_, name) => {
                        self.regions.insert(*name);
                    }

                    ty::BoundRegion::BrAnon(var) => {
                        self.regions.insert(InternedString::intern(&format!("'^{}", var)));
                    }

                    _ => (),
                }
            }

            _ => (),
        };

        r.super_visit_with(self)
    }
}

impl<'tcx> fmt::Display for traits::Goal<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::GoalKind::*;

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
                let mut collector = BoundNamesCollector::new();
                goal.skip_binder().visit_with(&mut collector);

                if !collector.is_empty() {
                    write!(fmt, "{}<", qkind)?;
                    collector.write_names(fmt)?;
                    write!(fmt, "> {{ ")?;
                }

                write!(fmt, "{}", goal.skip_binder())?;

                if !collector.is_empty() {
                    write!(fmt, " }}")?;
                }

                Ok(())
            }
            Subtype(a, b) => write!(fmt, "{} <: {}", a, b),
            CannotProve => write!(fmt, "CannotProve"),
        }
    }
}

impl<'tcx> fmt::Display for traits::ProgramClause<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let traits::ProgramClause { goal, hypotheses, .. } = self;
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
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::traits::Clause::*;

        match self {
            Implies(clause) => write!(fmt, "{}", clause),
            ForAll(clause) => {
                let mut collector = BoundNamesCollector::new();
                clause.skip_binder().visit_with(&mut collector);

                if !collector.is_empty() {
                    write!(fmt, "forall<")?;
                    collector.write_names(fmt)?;
                    write!(fmt, "> {{ ")?;
                }

                write!(fmt, "{}", clause.skip_binder())?;

                if !collector.is_empty() {
                    write!(fmt, " }}")?;
                }

                Ok(())
            }
        }
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
                tcx.lift(&(a, b)).and_then(|(a, b)|
                    tcx.lift(err)
                        .map(|err| super::OutputTypeParameterMismatch(a, b, err))
                )
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
            super::ReferenceOutlivesReferent(ty) => {
                tcx.lift(&ty).map(super::ReferenceOutlivesReferent)
            }
            super::ObjectTypeBound(ty, r) => tcx.lift(&ty).and_then(|ty|
                tcx.lift(&r)
                   .and_then(|r| Some(super::ObjectTypeBound(ty, r)))
            ),
            super::ObjectCastObligation(ty) => tcx.lift(&ty).map(super::ObjectCastObligation),
            super::AssignmentLhsSized => Some(super::AssignmentLhsSized),
            super::TupleInitializerSized => Some(super::TupleInitializerSized),
            super::StructInitializerSized => Some(super::StructInitializerSized),
            super::VariableType(id) => Some(super::VariableType(id)),
            super::ReturnType(id) => Some(super::ReturnType(id)),
            super::SizedArgumentType => Some(super::SizedArgumentType),
            super::SizedReturnType => Some(super::SizedReturnType),
            super::SizedYieldType => Some(super::SizedYieldType),
            super::RepeatVec => Some(super::RepeatVec),
            super::FieldSized { adt_kind, last } => Some(super::FieldSized { adt_kind, last }),
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
            super::MatchExpressionArm {
                arm_span,
                source,
                ref prior_arms,
                last_ty,
                discrim_hir_id,
            } => {
                tcx.lift(&last_ty).map(|last_ty| {
                    super::MatchExpressionArm {
                        arm_span,
                        source,
                        prior_arms: prior_arms.clone(),
                        last_ty,
                        discrim_hir_id,
                    }
                })
            }
            super::MatchExpressionArmPattern { span, ty } => {
                tcx.lift(&ty).map(|ty| super::MatchExpressionArmPattern { span, ty })
            }
            super::IfExpression { then, outer, semicolon } => Some(super::IfExpression {
                then,
                outer,
                semicolon,
            }),
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
        tcx.lift(&self.parent_trait_ref).and_then(|trait_ref|
            tcx.lift(&*self.parent_code)
               .map(|code| traits::DerivedObligationCause {
                   parent_trait_ref: trait_ref,
                   parent_code: Rc::new(code),
               })
        )
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
            traits::VtableImpl(traits::VtableImplData {
                impl_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs|
                traits::VtableImpl(traits::VtableImplData {
                    impl_def_id,
                    substs,
                    nested,
                })
            ),
            traits::VtableAutoImpl(t) => Some(traits::VtableAutoImpl(t)),
            traits::VtableGenerator(traits::VtableGeneratorData {
                generator_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs|
                traits::VtableGenerator(traits::VtableGeneratorData {
                    generator_def_id: generator_def_id,
                    substs: substs,
                    nested: nested,
                })
            ),
            traits::VtableClosure(traits::VtableClosureData {
                closure_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs|
                traits::VtableClosure(traits::VtableClosureData {
                    closure_def_id,
                    substs,
                    nested,
                })
            ),
            traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested }) => {
                tcx.lift(&fn_ty).map(|fn_ty|
                    traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, nested })
                )
            }
            traits::VtableParam(n) => Some(traits::VtableParam(n)),
            traits::VtableBuiltin(n) => Some(traits::VtableBuiltin(n)),
            traits::VtableObject(traits::VtableObjectData {
                upcast_trait_ref,
                vtable_base,
                nested,
            }) => tcx.lift(&upcast_trait_ref).map(|trait_ref|
                traits::VtableObject(traits::VtableObjectData {
                    upcast_trait_ref: trait_ref,
                    vtable_base,
                    nested,
                })
            ),
            traits::VtableTraitAlias(traits::VtableTraitAliasData {
                alias_def_id,
                substs,
                nested,
            }) => tcx.lift(&substs).map(|substs|
                traits::VtableTraitAlias(traits::VtableTraitAliasData {
                    alias_def_id,
                    substs,
                    nested,
                })
            ),
        }
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

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::WellFormed<'a> {
        type Lifted = traits::WellFormed<'tcx>;
        (traits::WellFormed::Trait)(trait_ref),
        (traits::WellFormed::Ty)(ty),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::FromEnv<'a> {
        type Lifted = traits::FromEnv<'tcx>;
        (traits::FromEnv::Trait)(trait_ref),
        (traits::FromEnv::Ty)(ty),
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

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for traits::GoalKind<'a> {
        type Lifted = traits::GoalKind<'tcx>;
        (traits::GoalKind::Implies)(hypotheses, goal),
        (traits::GoalKind::And)(goal1, goal2),
        (traits::GoalKind::Not)(goal),
        (traits::GoalKind::DomainGoal)(domain_goal),
        (traits::GoalKind::Quantified)(kind, goal),
        (traits::GoalKind::Subtype)(a, b),
        (traits::GoalKind::CannotProve),
    }
}

impl<'a, 'tcx> Lift<'tcx> for traits::Environment<'a> {
    type Lifted = traits::Environment<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.clauses).map(|clauses| {
            traits::Environment {
                clauses,
            }
        })
    }
}

impl<'a, 'tcx, G: Lift<'tcx>> Lift<'tcx> for traits::InEnvironment<'a, G> {
    type Lifted = traits::InEnvironment<'tcx, G::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.environment).and_then(|environment| {
            tcx.lift(&self.goal).map(|goal| {
                traits::InEnvironment {
                    environment,
                    goal,
                }
            })
        })
    }
}

impl<'tcx, C> Lift<'tcx> for chalk_engine::ExClause<C>
where
    C: chalk_engine::context::Context + Clone,
    C: traits::ChalkContextLift<'tcx>,
{
    type Lifted = C::LiftedExClause;

    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        <C as traits::ChalkContextLift>::lift_ex_clause_to_tcx(self, tcx)
    }
}

impl<'tcx, C> Lift<'tcx> for chalk_engine::DelayedLiteral<C>
where
    C: chalk_engine::context::Context + Clone,
    C: traits::ChalkContextLift<'tcx>,
{
    type Lifted = C::LiftedDelayedLiteral;

    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        <C as traits::ChalkContextLift>::lift_delayed_literal_to_tcx(self, tcx)
    }
}

impl<'tcx, C> Lift<'tcx> for chalk_engine::Literal<C>
where
    C: chalk_engine::context::Context + Clone,
    C: traits::ChalkContextLift<'tcx>,
{
    type Lifted = C::LiftedLiteral;

    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        <C as traits::ChalkContextLift>::lift_literal_to_tcx(self, tcx)
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.

impl<'tcx, O: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Obligation<'tcx, O> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
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

BraceStructTypeFoldableImpl! {
    impl<'tcx, N> TypeFoldable<'tcx> for traits::VtableTraitAliasData<'tcx, N> {
        alias_def_id, substs, nested
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
        (traits::VtableTraitAlias)(a),
    } where N: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for Normalized<'tcx, T> {
        value,
        obligations
    } where T: TypeFoldable<'tcx>
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::WhereClause<'tcx> {
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

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::FromEnv<'tcx> {
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

CloneTypeFoldableAndLiftImpls! {
    traits::QuantifierKind,
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::GoalKind<'tcx> {
        (traits::GoalKind::Implies)(hypotheses, goal),
        (traits::GoalKind::And)(goal1, goal2),
        (traits::GoalKind::Not)(goal),
        (traits::GoalKind::DomainGoal)(domain_goal),
        (traits::GoalKind::Quantified)(qkind, goal),
        (traits::GoalKind::Subtype)(a, b),
        (traits::GoalKind::CannotProve),
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<traits::Goal<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter()
            .map(|t| t.fold_with(folder))
            .collect::<SmallVec<[_; 8]>>();
        folder.tcx().intern_goals(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::Goal<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
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
        hypotheses,
        category,
    }
}

CloneTypeFoldableAndLiftImpls! {
    traits::ProgramClauseCategory,
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::Clause<'tcx> {
        (traits::Clause::Implies)(clause),
        (traits::Clause::ForAll)(clause),
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for traits::Environment<'tcx> { clauses }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, G> TypeFoldable<'tcx> for traits::InEnvironment<'tcx, G> {
        environment,
        goal
    } where G: TypeFoldable<'tcx>
}

impl<'tcx> TypeFoldable<'tcx> for traits::Clauses<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter()
            .map(|t| t.fold_with(folder))
            .collect::<SmallVec<[_; 8]>>();
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
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
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
