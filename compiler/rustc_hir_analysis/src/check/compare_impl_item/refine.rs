use itertools::Itertools as _;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint_defs::builtin::{REFINING_IMPL_TRAIT_INTERNAL, REFINING_IMPL_TRAIT_REACHABLE};
use rustc_middle::span_bug;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor, TypingMode,
};
use rustc_span::Span;
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::{ObligationCtxt, elaborate, normalize_param_env_or_error};

/// Check that an implementation does not refine an RPITIT from a trait method signature.
pub(crate) fn check_refining_return_position_impl_trait_in_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) {
    if !tcx.impl_method_has_trait_impl_trait_tys(impl_m.def_id) {
        return;
    }

    // unreachable traits don't have any library guarantees, there's no need to do this check.
    let is_internal = trait_m
        .container_id(tcx)
        .as_local()
        .is_some_and(|trait_def_id| !tcx.effective_visibilities(()).is_reachable(trait_def_id))
        // If a type in the trait ref is private, then there's also no reason to do this check.
        || impl_trait_ref.args.iter().any(|arg| {
            if let Some(ty) = arg.as_type()
                && let Some(self_visibility) = type_visibility(tcx, ty)
            {
                return !self_visibility.is_public();
            }
            false
        });

    let impl_def_id = impl_m.container_id(tcx);
    let impl_m_args = ty::GenericArgs::identity_for_item(tcx, impl_m.def_id);
    let trait_m_to_impl_m_args = impl_m_args.rebase_onto(tcx, impl_def_id, impl_trait_ref.args);
    let bound_trait_m_sig = tcx.fn_sig(trait_m.def_id).instantiate(tcx, trait_m_to_impl_m_args);
    let trait_m_sig = tcx.liberate_late_bound_regions(impl_m.def_id, bound_trait_m_sig);
    // replace the self type of the trait ref with `Self` so that diagnostics render better.
    let trait_m_sig_with_self_for_diag = tcx.liberate_late_bound_regions(
        impl_m.def_id,
        tcx.fn_sig(trait_m.def_id).instantiate(
            tcx,
            tcx.mk_args_from_iter(
                [tcx.types.self_param.into()]
                    .into_iter()
                    .chain(trait_m_to_impl_m_args.iter().skip(1)),
            ),
        ),
    );

    let Ok(hidden_tys) = tcx.collect_return_position_impl_trait_in_trait_tys(impl_m.def_id) else {
        // Error already emitted, no need to delay another.
        return;
    };

    if hidden_tys.items().any(|(_, &ty)| ty.skip_binder().references_error()) {
        return;
    }

    let mut collector = ImplTraitInTraitCollector { tcx, types: FxIndexSet::default() };
    trait_m_sig.visit_with(&mut collector);

    // Bound that we find on RPITITs in the trait signature.
    let mut trait_bounds = vec![];
    // Bounds that we find on the RPITITs in the impl signature.
    let mut impl_bounds = vec![];
    // Pairs of trait and impl opaques.
    let mut pairs = vec![];

    for trait_projection in collector.types.into_iter().rev() {
        let impl_opaque_args = trait_projection.args.rebase_onto(tcx, trait_m.def_id, impl_m_args);
        let hidden_ty = hidden_tys[&trait_projection.def_id].instantiate(tcx, impl_opaque_args);

        // If the hidden type is not an opaque, then we have "refined" the trait signature.
        let ty::Alias(ty::Opaque, impl_opaque) = *hidden_ty.kind() else {
            report_mismatched_rpitit_signature(
                tcx,
                trait_m_sig_with_self_for_diag,
                trait_m.def_id,
                impl_m.def_id,
                None,
                is_internal,
            );
            return;
        };

        // This opaque also needs to be from the impl method -- otherwise,
        // it's a refinement to a TAIT.
        if !tcx.hir_get_if_local(impl_opaque.def_id).is_some_and(|node| {
            matches!(
                node.expect_opaque_ty().origin,
                hir::OpaqueTyOrigin::AsyncFn { parent, .. }  | hir::OpaqueTyOrigin::FnReturn { parent, .. }
                    if parent == impl_m.def_id.expect_local()
            )
        }) {
            report_mismatched_rpitit_signature(
                tcx,
                trait_m_sig_with_self_for_diag,
                trait_m.def_id,
                impl_m.def_id,
                None,
                is_internal,
            );
            return;
        }

        trait_bounds.extend(
            tcx.item_bounds(trait_projection.def_id).iter_instantiated(tcx, trait_projection.args),
        );
        impl_bounds.extend(elaborate(
            tcx,
            tcx.explicit_item_bounds(impl_opaque.def_id)
                .iter_instantiated_copied(tcx, impl_opaque.args),
        ));

        pairs.push((trait_projection, impl_opaque));
    }

    let hybrid_preds = tcx
        .predicates_of(impl_def_id)
        .instantiate_identity(tcx)
        .into_iter()
        .chain(tcx.predicates_of(trait_m.def_id).instantiate_own(tcx, trait_m_to_impl_m_args))
        .map(|(clause, _)| clause);
    let param_env = ty::ParamEnv::new(tcx.mk_clauses_from_iter(hybrid_preds));
    let param_env = normalize_param_env_or_error(tcx, param_env, ObligationCause::dummy());

    let ref infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new(infcx);

    // Normalize the bounds. This has two purposes:
    //
    // 1. Project the RPITIT projections from the trait to the opaques on the impl,
    //    which means that they don't need to be mapped manually.
    //
    // 2. Deeply normalize any other projections that show up in the bound. That makes sure
    //    that we don't consider `tests/ui/async-await/in-trait/async-associated-types.rs`
    //    or `tests/ui/impl-trait/in-trait/refine-normalize.rs` to be refining.
    let Ok((trait_bounds, impl_bounds)) =
        ocx.deeply_normalize(&ObligationCause::dummy(), param_env, (trait_bounds, impl_bounds))
    else {
        tcx.dcx().delayed_bug("encountered errors when checking RPITIT refinement (selection)");
        return;
    };

    // Since we've normalized things, we need to resolve regions, since we'll
    // possibly have introduced region vars during projection. We don't expect
    // this resolution to have incurred any region errors -- but if we do, then
    // just delay a bug.
    let mut implied_wf_types = FxIndexSet::default();
    implied_wf_types.extend(trait_m_sig.inputs_and_output);
    implied_wf_types.extend(ocx.normalize(
        &ObligationCause::dummy(),
        param_env,
        trait_m_sig.inputs_and_output,
    ));
    if !ocx.select_all_or_error().is_empty() {
        tcx.dcx().delayed_bug("encountered errors when checking RPITIT refinement (selection)");
        return;
    }
    let errors = infcx.resolve_regions(impl_m.def_id.expect_local(), param_env, implied_wf_types);
    if !errors.is_empty() {
        tcx.dcx().delayed_bug("encountered errors when checking RPITIT refinement (regions)");
        return;
    }
    // Resolve any lifetime variables that may have been introduced during normalization.
    let Ok((trait_bounds, impl_bounds)) = infcx.fully_resolve((trait_bounds, impl_bounds)) else {
        // If resolution didn't fully complete, we cannot continue checking RPITIT refinement, and
        // delay a bug as the original code contains load-bearing errors.
        tcx.dcx().delayed_bug("encountered errors when checking RPITIT refinement (resolution)");
        return;
    };

    if trait_bounds.references_error() || impl_bounds.references_error() {
        return;
    }

    // For quicker lookup, use an `IndexSet` (we don't use one earlier because
    // it's not foldable..).
    // Also, We have to anonymize binders in these types because they may contain
    // `BrNamed` bound vars, which contain unique `DefId`s which correspond to syntax
    // locations that we don't care about when checking bound equality.
    let trait_bounds = FxIndexSet::from_iter(trait_bounds.fold_with(&mut Anonymize { tcx }));
    let impl_bounds = impl_bounds.fold_with(&mut Anonymize { tcx });

    // Find any clauses that are present in the impl's RPITITs that are not
    // present in the trait's RPITITs. This will trigger on trivial predicates,
    // too, since we *do not* use the trait solver to prove that the RPITIT's
    // bounds are not stronger -- we're doing a simple, syntactic compatibility
    // check between bounds. This is strictly forwards compatible, though.
    for (clause, span) in impl_bounds {
        if !trait_bounds.contains(&clause) {
            report_mismatched_rpitit_signature(
                tcx,
                trait_m_sig_with_self_for_diag,
                trait_m.def_id,
                impl_m.def_id,
                Some(span),
                is_internal,
            );
            return;
        }
    }

    // Make sure that the RPITIT doesn't capture fewer regions than
    // the trait definition. We hard-error if it captures *more*, since that
    // is literally unrepresentable in the type system; however, we may be
    // promising stronger outlives guarantees if we capture *fewer* regions.
    for (trait_projection, impl_opaque) in pairs {
        let impl_variances = tcx.variances_of(impl_opaque.def_id);
        let impl_captures: FxIndexSet<_> = impl_opaque
            .args
            .iter()
            .zip_eq(impl_variances)
            .filter(|(_, v)| **v == ty::Invariant)
            .map(|(arg, _)| arg)
            .collect();

        let trait_variances = tcx.variances_of(trait_projection.def_id);
        let mut trait_captures = FxIndexSet::default();
        for (arg, variance) in trait_projection.args.iter().zip_eq(trait_variances) {
            if *variance != ty::Invariant {
                continue;
            }
            arg.visit_with(&mut CollectParams { params: &mut trait_captures });
        }

        if !trait_captures.iter().all(|arg| impl_captures.contains(arg)) {
            report_mismatched_rpitit_captures(
                tcx,
                impl_opaque.def_id.expect_local(),
                trait_captures,
                is_internal,
            );
        }
    }
}

struct ImplTraitInTraitCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: FxIndexSet<ty::AliasTy<'tcx>>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ImplTraitInTraitCollector<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if let ty::Alias(ty::Projection, proj) = *ty.kind()
            && self.tcx.is_impl_trait_in_trait(proj.def_id)
        {
            if self.types.insert(proj) {
                for (pred, _) in self
                    .tcx
                    .explicit_item_bounds(proj.def_id)
                    .iter_instantiated_copied(self.tcx, proj.args)
                {
                    pred.visit_with(self);
                }
            }
        } else {
            ty.super_visit_with(self);
        }
    }
}

fn report_mismatched_rpitit_signature<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_m_sig: ty::FnSig<'tcx>,
    trait_m_def_id: DefId,
    impl_m_def_id: DefId,
    unmatched_bound: Option<Span>,
    is_internal: bool,
) {
    let mapping = std::iter::zip(
        tcx.fn_sig(trait_m_def_id).skip_binder().bound_vars(),
        tcx.fn_sig(impl_m_def_id).skip_binder().bound_vars(),
    )
    .enumerate()
    .filter_map(|(idx, (impl_bv, trait_bv))| {
        if let ty::BoundVariableKind::Region(impl_bv) = impl_bv
            && let ty::BoundVariableKind::Region(trait_bv) = trait_bv
        {
            let var = ty::BoundVar::from_usize(idx);
            Some((
                ty::LateParamRegionKind::from_bound(var, impl_bv),
                ty::LateParamRegionKind::from_bound(var, trait_bv),
            ))
        } else {
            None
        }
    })
    .collect();

    let mut return_ty = trait_m_sig.output().fold_with(&mut super::RemapLateParam { tcx, mapping });

    if tcx.asyncness(impl_m_def_id).is_async() && tcx.asyncness(trait_m_def_id).is_async() {
        let ty::Alias(ty::Projection, future_ty) = return_ty.kind() else {
            span_bug!(
                tcx.def_span(trait_m_def_id),
                "expected return type of async fn in trait to be a AFIT projection"
            );
        };
        let Some(future_output_ty) = tcx
            .explicit_item_bounds(future_ty.def_id)
            .iter_instantiated_copied(tcx, future_ty.args)
            .find_map(|(clause, _)| match clause.kind().no_bound_vars()? {
                ty::ClauseKind::Projection(proj) => proj.term.as_type(),
                _ => None,
            })
        else {
            span_bug!(tcx.def_span(trait_m_def_id), "expected `Future` projection bound in AFIT");
        };
        return_ty = future_output_ty;
    }

    let (span, impl_return_span, pre, post) =
        match tcx.hir_node_by_def_id(impl_m_def_id.expect_local()).fn_decl().unwrap().output {
            hir::FnRetTy::DefaultReturn(span) => (tcx.def_span(impl_m_def_id), span, "-> ", " "),
            hir::FnRetTy::Return(ty) => (ty.span, ty.span, "", ""),
        };
    let trait_return_span =
        tcx.hir_get_if_local(trait_m_def_id).map(|node| match node.fn_decl().unwrap().output {
            hir::FnRetTy::DefaultReturn(_) => tcx.def_span(trait_m_def_id),
            hir::FnRetTy::Return(ty) => ty.span,
        });

    let span = unmatched_bound.unwrap_or(span);
    tcx.emit_node_span_lint(
        if is_internal { REFINING_IMPL_TRAIT_INTERNAL } else { REFINING_IMPL_TRAIT_REACHABLE },
        tcx.local_def_id_to_hir_id(impl_m_def_id.expect_local()),
        span,
        crate::errors::ReturnPositionImplTraitInTraitRefined {
            impl_return_span,
            trait_return_span,
            pre,
            post,
            return_ty,
            unmatched_bound,
        },
    );
}

fn type_visibility<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ty::Visibility<DefId>> {
    match *ty.kind() {
        ty::Ref(_, ty, _) => type_visibility(tcx, ty),
        ty::Adt(def, args) => {
            if def.is_fundamental() {
                type_visibility(tcx, args.type_at(0))
            } else {
                Some(tcx.visibility(def.did()))
            }
        }
        _ => None,
    }
}

struct Anonymize<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for Anonymize<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.tcx.anonymize_bound_vars(t)
    }
}

struct CollectParams<'a, 'tcx> {
    params: &'a mut FxIndexSet<ty::GenericArg<'tcx>>,
}
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for CollectParams<'_, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if let ty::Param(_) = ty.kind() {
            self.params.insert(ty.into());
        } else {
            ty.super_visit_with(self);
        }
    }
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        match r.kind() {
            ty::ReEarlyParam(_) | ty::ReLateParam(_) => {
                self.params.insert(r.into());
            }
            _ => {}
        }
    }
    fn visit_const(&mut self, ct: ty::Const<'tcx>) {
        if let ty::ConstKind::Param(_) = ct.kind() {
            self.params.insert(ct.into());
        } else {
            ct.super_visit_with(self);
        }
    }
}

fn report_mismatched_rpitit_captures<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_opaque_def_id: LocalDefId,
    mut trait_captured_args: FxIndexSet<ty::GenericArg<'tcx>>,
    is_internal: bool,
) {
    let Some(use_bound_span) =
        tcx.hir_node_by_def_id(impl_opaque_def_id).expect_opaque_ty().bounds.iter().find_map(
            |bound| match *bound {
                rustc_hir::GenericBound::Use(_, span) => Some(span),
                hir::GenericBound::Trait(_) | hir::GenericBound::Outlives(_) => None,
            },
        )
    else {
        // I have no idea when you would ever undercapture without a `use<..>`.
        tcx.dcx().delayed_bug("expected use<..> to undercapture in an impl opaque");
        return;
    };

    trait_captured_args
        .sort_by_cached_key(|arg| !matches!(arg.kind(), ty::GenericArgKind::Lifetime(_)));
    let suggestion = format!("use<{}>", trait_captured_args.iter().join(", "));

    tcx.emit_node_span_lint(
        if is_internal { REFINING_IMPL_TRAIT_INTERNAL } else { REFINING_IMPL_TRAIT_REACHABLE },
        tcx.local_def_id_to_hir_id(impl_opaque_def_id),
        use_bound_span,
        crate::errors::ReturnPositionImplTraitInTraitRefinedLifetimes {
            suggestion_span: use_bound_span,
            suggestion,
        },
    );
}
