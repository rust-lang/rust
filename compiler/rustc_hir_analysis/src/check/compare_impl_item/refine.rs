use std::ops::ControlFlow;

use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::Obligation;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFolder, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::ErrorGuaranteed;
use rustc_span::{sym, Span};
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_type_ir::fold::TypeFoldable;

/// Check that an implementation does not refine an RPITIT from a trait method signature.
pub(super) fn compare_impl_trait_in_trait_predicate_entailment<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    if !tcx.impl_method_has_trait_impl_trait_tys(impl_m.def_id)
        || tcx.has_attr(impl_m.def_id, sym::refine)
    {
        return Ok(());
    }

    let hidden_tys = tcx.collect_return_position_impl_trait_in_trait_tys(impl_m.def_id)?;

    let impl_def_id = impl_m.container_id(tcx);
    //let trait_def_id = trait_m.container_id(tcx);
    let trait_m_to_impl_m_substs = ty::InternalSubsts::identity_for_item(tcx, impl_m.def_id)
        .rebase_onto(tcx, impl_def_id, impl_trait_ref.substs);

    let bound_trait_m_sig = tcx.fn_sig(trait_m.def_id).subst(tcx, trait_m_to_impl_m_substs);
    let trait_m_sig = tcx.liberate_late_bound_regions(impl_m.def_id, bound_trait_m_sig);

    let mut visitor = ImplTraitInTraitCollector { tcx, types: FxIndexMap::default() };
    trait_m_sig.visit_with(&mut visitor);

    let mut reverse_mapping = FxIndexMap::default();
    let mut bounds_to_prove = vec![];
    for (rpitit_def_id, rpitit_substs) in visitor.types {
        let hidden_ty = hidden_tys
            .get(&rpitit_def_id)
            .expect("expected hidden type for RPITIT")
            .subst_identity();
        reverse_mapping.insert(hidden_ty, tcx.mk_projection(rpitit_def_id, rpitit_substs));

        let ty::Alias(ty::Opaque, opaque_ty) = *hidden_ty.kind() else {
            return Err(report_mismatched_rpitit_signature(
                tcx,
                trait_m_sig,
                trait_m.def_id,
                impl_m.def_id,
                None,
            ));
        };

        // Check that this is an opaque that comes from our impl fn
        if !tcx.hir().get_if_local(opaque_ty.def_id).map_or(false, |node| {
            matches!(
                node.expect_item().expect_opaque_ty().origin,
                hir::OpaqueTyOrigin::AsyncFn(def_id)  | hir::OpaqueTyOrigin::FnReturn(def_id)
                    if def_id == impl_m.def_id.expect_local()
            )
        }) {
            return Err(report_mismatched_rpitit_signature(
                tcx,
                trait_m_sig,
                trait_m.def_id,
                impl_m.def_id,
                None,
            ));
        }

        bounds_to_prove.extend(
            tcx.explicit_item_bounds(opaque_ty.def_id)
                .iter_instantiated_copied(tcx, opaque_ty.args),
        );
    }

    let infcx = tcx.infer_ctxt().build();
    let ocx = ObligationCtxt::new(&infcx);
    let param_env =
        tcx.param_env(impl_m.def_id).with_hidden_return_position_impl_trait_in_trait_tys();

    ocx.register_obligations(
        bounds_to_prove.fold_with(&mut ReverseMapper { tcx, reverse_mapping }).into_iter().map(
            |(pred, span)| {
                Obligation::new(tcx, ObligationCause::dummy_with_span(span), param_env, pred)
            },
        ),
    );

    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let span = errors.first().unwrap().obligation.cause.span;
        return Err(report_mismatched_rpitit_signature(
            tcx,
            trait_m_sig,
            trait_m.def_id,
            impl_m.def_id,
            Some(span),
        ));
    }

    Ok(())
}

struct ImplTraitInTraitCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: FxIndexMap<DefId, ty::GenericArgsRef<'tcx>>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ImplTraitInTraitCollector<'tcx> {
    type BreakTy = !;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> std::ops::ControlFlow<Self::BreakTy> {
        if let ty::Alias(ty::Projection, proj) = *ty.kind()
            && self.tcx.is_impl_trait_in_trait(proj.def_id)
        {
            if self.types.insert(proj.def_id, proj.args).is_none() {
                for (pred, _) in self
                    .tcx
                    .explicit_item_bounds(proj.def_id)
                    .iter_instantiated_copied(self.tcx, proj.args)
                {
                    pred.visit_with(self)?;
                }
            }
            ControlFlow::Continue(())
        } else {
            ty.super_visit_with(self)
        }
    }
}

struct ReverseMapper<'tcx> {
    tcx: TyCtxt<'tcx>,
    reverse_mapping: FxIndexMap<Ty<'tcx>, Ty<'tcx>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReverseMapper<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let Some(ty) = self.reverse_mapping.get(&ty) { *ty } else { ty.super_fold_with(self) }
    }
}

fn report_mismatched_rpitit_signature<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_m_sig: ty::FnSig<'tcx>,
    trait_m_def_id: DefId,
    impl_m_def_id: DefId,
    unmatched_bound: Option<Span>,
) -> ErrorGuaranteed {
    let mapping = std::iter::zip(
        tcx.fn_sig(trait_m_def_id).skip_binder().bound_vars(),
        tcx.fn_sig(impl_m_def_id).skip_binder().bound_vars(),
    )
    .filter_map(|(impl_bv, trait_bv)| {
        if let ty::BoundVariableKind::Region(impl_bv) = impl_bv
            && let ty::BoundVariableKind::Region(trait_bv) = trait_bv
        {
            Some((impl_bv, trait_bv))
        } else {
            None
        }
    })
    .collect();

    let return_ty =
        trait_m_sig.output().fold_with(&mut super::RemapLateBound { tcx, mapping: &mapping });

    let (span, impl_return_span, sugg) =
        match tcx.hir().get_by_def_id(impl_m_def_id.expect_local()).fn_decl().unwrap().output {
            hir::FnRetTy::DefaultReturn(span) => {
                (tcx.def_span(impl_m_def_id), span, format!("-> {return_ty} "))
            }
            hir::FnRetTy::Return(ty) => (ty.span, ty.span, format!("{return_ty}")),
        };
    let trait_return_span =
        tcx.hir().get_if_local(trait_m_def_id).map(|node| match node.fn_decl().unwrap().output {
            hir::FnRetTy::DefaultReturn(_) => tcx.def_span(trait_m_def_id),
            hir::FnRetTy::Return(ty) => ty.span,
        });

    tcx.sess.emit_err(crate::errors::ReturnPositionImplTraitInTraitRefined {
        span,
        impl_return_span,
        trait_return_span,
        sugg,
        unmatched_bound,
    })
}
