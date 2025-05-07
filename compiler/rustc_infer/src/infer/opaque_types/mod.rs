use hir::def_id::{DefId, LocalDefId};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_middle::bug;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{
    self, BottomUpFolder, OpaqueHiddenType, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable,
    TypeVisitableExt,
};
use rustc_span::Span;
use tracing::{debug, instrument};

use super::{DefineOpaqueTypes, RegionVariableOrigin};
use crate::errors::OpaqueHiddenTypeDiag;
use crate::infer::{InferCtxt, InferOk};
use crate::traits::{self, Obligation, PredicateObligations};

mod table;

pub(crate) type OpaqueTypeMap<'tcx> = FxIndexMap<OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>>;
pub(crate) use table::{OpaqueTypeStorage, OpaqueTypeTable};

impl<'tcx> InferCtxt<'tcx> {
    /// This is a backwards compatibility hack to prevent breaking changes from
    /// lazy TAIT around RPIT handling.
    pub fn replace_opaque_types_with_inference_vars<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        value: T,
        body_id: LocalDefId,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
    ) -> InferOk<'tcx, T> {
        // We handle opaque types differently in the new solver.
        if self.next_trait_solver() {
            return InferOk { value, obligations: PredicateObligations::new() };
        }

        if !value.has_opaque_types() {
            return InferOk { value, obligations: PredicateObligations::new() };
        }

        let mut obligations = PredicateObligations::new();
        let value = value.fold_with(&mut BottomUpFolder {
            tcx: self.tcx,
            lt_op: |lt| lt,
            ct_op: |ct| ct,
            ty_op: |ty| match *ty.kind() {
                ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. })
                    if self.can_define_opaque_ty(def_id) && !ty.has_escaping_bound_vars() =>
                {
                    let def_span = self.tcx.def_span(def_id);
                    let span = if span.contains(def_span) { def_span } else { span };
                    let ty_var = self.next_ty_var(span);
                    obligations.extend(
                        self.handle_opaque_type(ty, ty_var, span, param_env)
                            .unwrap()
                            .into_iter()
                            .map(|goal| {
                                Obligation::new(
                                    self.tcx,
                                    ObligationCause::new(
                                        span,
                                        body_id,
                                        traits::ObligationCauseCode::OpaqueReturnType(None),
                                    ),
                                    goal.param_env,
                                    goal.predicate,
                                )
                            }),
                    );
                    ty_var
                }
                _ => ty,
            },
        });
        InferOk { value, obligations }
    }

    pub fn handle_opaque_type(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, TypeError<'tcx>> {
        debug_assert!(!self.next_trait_solver());
        let process = |a: Ty<'tcx>, b: Ty<'tcx>| match *a.kind() {
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) if def_id.is_local() => {
                let def_id = def_id.expect_local();
                if let ty::TypingMode::Coherence = self.typing_mode() {
                    // See comment on `insert_hidden_type` for why this is sufficient in coherence
                    return Some(self.register_hidden_type(
                        OpaqueTypeKey { def_id, args },
                        span,
                        param_env,
                        b,
                    ));
                }
                // Check that this is `impl Trait` type is
                // declared by `parent_def_id` -- i.e., one whose
                // value we are inferring. At present, this is
                // always true during the first phase of
                // type-check, but not always true later on during
                // NLL. Once we support named opaque types more fully,
                // this same scenario will be able to arise during all phases.
                //
                // Here is an example using type alias `impl Trait`
                // that indicates the distinction we are checking for:
                //
                // ```rust
                // mod a {
                //   pub type Foo = impl Iterator;
                //   pub fn make_foo() -> Foo { .. }
                // }
                //
                // mod b {
                //   fn foo() -> a::Foo { a::make_foo() }
                // }
                // ```
                //
                // Here, the return type of `foo` references an
                // `Opaque` indeed, but not one whose value is
                // presently being inferred. You can get into a
                // similar situation with closure return types
                // today:
                //
                // ```rust
                // fn foo() -> impl Iterator { .. }
                // fn bar() {
                //     let x = || foo(); // returns the Opaque assoc with `foo`
                // }
                // ```
                if !self.can_define_opaque_ty(def_id) {
                    return None;
                }

                if let ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }) = *b.kind() {
                    // We could accept this, but there are various ways to handle this situation,
                    // and we don't want to make a decision on it right now. Likely this case is so
                    // super rare anyway, that no one encounters it in practice. It does occur
                    // however in `fn fut() -> impl Future<Output = i32> { async { 42 } }`, where
                    // it is of no concern, so we only check for TAITs.
                    if self.can_define_opaque_ty(b_def_id)
                        && matches!(
                            self.tcx.opaque_ty_origin(b_def_id),
                            hir::OpaqueTyOrigin::TyAlias { .. }
                        )
                    {
                        self.dcx().emit_err(OpaqueHiddenTypeDiag {
                            span,
                            hidden_type: self.tcx.def_span(b_def_id),
                            opaque_type: self.tcx.def_span(def_id),
                        });
                    }
                }
                Some(self.register_hidden_type(OpaqueTypeKey { def_id, args }, span, param_env, b))
            }
            _ => None,
        };
        if let Some(res) = process(a, b) {
            res
        } else if let Some(res) = process(b, a) {
            res
        } else {
            let (a, b) = self.resolve_vars_if_possible((a, b));
            Err(TypeError::Sorts(ExpectedFound::new(a, b)))
        }
    }
}

impl<'tcx> InferCtxt<'tcx> {
    #[instrument(skip(self), level = "debug")]
    fn register_hidden_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, TypeError<'tcx>> {
        let mut goals = Vec::new();

        self.insert_hidden_type(opaque_type_key, span, param_env, hidden_ty, &mut goals)?;

        self.add_item_bounds_for_hidden_type(
            opaque_type_key.def_id.to_def_id(),
            opaque_type_key.args,
            param_env,
            hidden_ty,
            &mut goals,
        );

        Ok(goals)
    }

    /// Insert a hidden type into the opaque type storage, making sure
    /// it hasn't previously been defined. This does not emit any
    /// constraints and it's the responsibility of the caller to make
    /// sure that the item bounds of the opaque are checked.
    pub fn register_hidden_type_in_storage(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        hidden_ty: OpaqueHiddenType<'tcx>,
    ) -> Option<Ty<'tcx>> {
        self.inner.borrow_mut().opaque_types().register(opaque_type_key, hidden_ty)
    }

    /// Insert a hidden type into the opaque type storage, equating it
    /// with any previous entries if necessary.
    ///
    /// This **does not** add the item bounds of the opaque as nested
    /// obligations. That is only necessary when normalizing the opaque
    /// itself, not when getting the opaque type constraints from
    /// somewhere else.
    pub fn insert_hidden_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        goals: &mut Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> Result<(), TypeError<'tcx>> {
        let tcx = self.tcx;
        // Ideally, we'd get the span where *this specific `ty` came
        // from*, but right now we just use the span from the overall
        // value being folded. In simple cases like `-> impl Foo`,
        // these are the same span, but not in cases like `-> (impl
        // Foo, impl Bar)`.
        match self.typing_mode() {
            ty::TypingMode::Coherence => {
                // During intercrate we do not define opaque types but instead always
                // force ambiguity unless the hidden type is known to not implement
                // our trait.
                goals.push(Goal::new(tcx, param_env, ty::PredicateKind::Ambiguous));
            }
            ty::TypingMode::Analysis { .. } => {
                let prev = self
                    .inner
                    .borrow_mut()
                    .opaque_types()
                    .register(opaque_type_key, OpaqueHiddenType { ty: hidden_ty, span });
                if let Some(prev) = prev {
                    goals.extend(
                        self.at(&ObligationCause::dummy_with_span(span), param_env)
                            .eq(DefineOpaqueTypes::Yes, prev, hidden_ty)?
                            .obligations
                            .into_iter()
                            .map(|obligation| obligation.as_goal()),
                    );
                }
            }
            ty::TypingMode::Borrowck { .. } => {
                let prev = self
                    .inner
                    .borrow_mut()
                    .opaque_types()
                    .register(opaque_type_key, OpaqueHiddenType { ty: hidden_ty, span });

                // We either equate the new hidden type with the previous entry or with the type
                // inferred by HIR typeck.
                let actual = prev.unwrap_or_else(|| {
                    let actual = tcx
                        .type_of_opaque_hir_typeck(opaque_type_key.def_id)
                        .instantiate(self.tcx, opaque_type_key.args);
                    let actual = ty::fold_regions(tcx, actual, |re, _dbi| match re.kind() {
                        ty::ReErased => {
                            self.next_region_var(RegionVariableOrigin::MiscVariable(span))
                        }
                        _ => re,
                    });
                    actual
                });

                goals.extend(
                    self.at(&ObligationCause::dummy_with_span(span), param_env)
                        .eq(DefineOpaqueTypes::Yes, hidden_ty, actual)?
                        .obligations
                        .into_iter()
                        .map(|obligation| obligation.as_goal()),
                );
            }
            mode @ (ty::TypingMode::PostBorrowckAnalysis { .. } | ty::TypingMode::PostAnalysis) => {
                bug!("insert hidden type in {mode:?}")
            }
        }

        Ok(())
    }

    pub fn add_item_bounds_for_hidden_type(
        &self,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        goals: &mut Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) {
        let tcx = self.tcx;
        // Require that the hidden type is well-formed. We have to
        // make sure we wf-check the hidden type to fix #114728.
        //
        // However, we don't check that all types are well-formed.
        // We only do so for types provided by the user or if they are
        // "used", e.g. for method selection.
        //
        // This means we never check the wf requirements of the hidden
        // type during MIR borrowck, causing us to infer the wrong
        // lifetime for its member constraints which then results in
        // unexpected region errors.
        goals.push(Goal::new(tcx, param_env, ty::ClauseKind::WellFormed(hidden_ty.into())));

        let replace_opaques_in = |clause: ty::Clause<'tcx>, goals: &mut Vec<_>| {
            clause.fold_with(&mut BottomUpFolder {
                tcx,
                ty_op: |ty| match *ty.kind() {
                    // We can't normalize associated types from `rustc_infer`,
                    // but we can eagerly register inference variables for them.
                    // FIXME(RPITIT): Don't replace RPITITs with inference vars.
                    // FIXME(inherent_associated_types): Extend this to support `ty::Inherent`, too.
                    ty::Alias(ty::Projection, projection_ty)
                        if !projection_ty.has_escaping_bound_vars()
                            && !tcx.is_impl_trait_in_trait(projection_ty.def_id)
                            && !self.next_trait_solver() =>
                    {
                        let ty_var = self.next_ty_var(self.tcx.def_span(projection_ty.def_id));
                        goals.push(Goal::new(
                            self.tcx,
                            param_env,
                            ty::PredicateKind::Clause(ty::ClauseKind::Projection(
                                ty::ProjectionPredicate {
                                    projection_term: projection_ty.into(),
                                    term: ty_var.into(),
                                },
                            )),
                        ));
                        ty_var
                    }
                    // Replace all other mentions of the same opaque type with the hidden type,
                    // as the bounds must hold on the hidden type after all.
                    ty::Alias(ty::Opaque, ty::AliasTy { def_id: def_id2, args: args2, .. })
                        if def_id == def_id2 && args == args2 =>
                    {
                        hidden_ty
                    }
                    _ => ty,
                },
                lt_op: |lt| lt,
                ct_op: |ct| ct,
            })
        };

        let item_bounds = tcx.explicit_item_bounds(def_id);
        for (predicate, _) in item_bounds.iter_instantiated_copied(tcx, args) {
            let predicate = replace_opaques_in(predicate, goals);

            // Require that the predicate holds for the concrete type.
            debug!(?predicate);
            goals.push(Goal::new(self.tcx, param_env, predicate));
        }

        // If this opaque is being defined and it's conditionally const,
        if self.tcx.is_conditionally_const(def_id) {
            let item_bounds = tcx.explicit_implied_const_bounds(def_id);
            for (predicate, _) in item_bounds.iter_instantiated_copied(tcx, args) {
                let predicate = replace_opaques_in(
                    predicate.to_host_effect_clause(self.tcx, ty::BoundConstness::Maybe),
                    goals,
                );

                // Require that the predicate holds for the concrete type.
                debug!(?predicate);
                goals.push(Goal::new(self.tcx, param_env, predicate));
            }
        }
    }
}
