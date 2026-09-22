use rustc_errors::codes::{E0581, E0582};
use rustc_errors::struct_span_code_err;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, Node};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::Span;
use rustc_trait_selection::traits::ObligationCause;
use rustc_trait_selection::traits::bound_regions::{
    OutputTypeDependency, projection_output_dependency, unconstrained_output_regions,
    unconstrained_output_regions_after_normalization,
};

#[derive(Clone, Copy, Debug)]
pub struct LateBoundRegionCheck<'tcx> {
    pub dependency: OutputTypeDependency<'tcx>,
    pub span: Span,
    pub associated_item: Option<DefId>,
    pub supertrait_span: Option<Span>,
}

impl<'tcx> LateBoundRegionCheck<'tcx> {
    pub fn function(tcx: TyCtxt<'tcx>, signature: ty::PolyFnSig<'tcx>, span: Span) -> Self {
        Self {
            dependency: signature.map_bound(|signature| {
                (
                    tcx.mk_args_from_iter(
                        signature.inputs().iter().map(|&ty| ty::GenericArg::from(ty)),
                    ),
                    signature.output().into(),
                )
            }),
            span,
            associated_item: None,
            supertrait_span: None,
        }
    }

    pub fn projection(projection: ty::PolyProjectionClause<'tcx>, span: Span) -> Self {
        Self {
            dependency: projection_output_dependency(projection),
            span,
            associated_item: Some(projection.item_def_id()),
            supertrait_span: None,
        }
    }

    pub fn needs_context(&self, tcx: TyCtxt<'tcx>) -> bool {
        tcx.next_trait_solver_globally()
            && !self.dependency.references_error()
            && self.dependency.has_aliases()
            && !unconstrained_output_regions(tcx, self.dependency).is_empty()
    }

    pub fn check(&self, tcx: TyCtxt<'tcx>, owner: LocalDefId, param_env: ty::ParamEnv<'tcx>) {
        let remaining = unconstrained_output_regions_after_normalization(
            tcx,
            &ObligationCause::misc(self.span, owner),
            param_env,
            self.dependency,
        );
        for br in remaining {
            let span = self.output_span(tcx, br);
            let br_name = if let Some(name) = br.get_name(tcx) {
                format!("lifetime `{name}`")
            } else {
                "an anonymous lifetime".to_string()
            };
            let mut err = if let Some(item) = self.associated_item {
                struct_span_code_err!(
                    tcx.dcx(),
                    span,
                    E0582,
                    "binding for associated type `{}` references {}, \
                     which does not appear in the trait input types",
                    tcx.item_name(item),
                    br_name,
                )
            } else {
                struct_span_code_err!(
                    tcx.dcx(),
                    span,
                    E0581,
                    "return type references {}, which is not constrained by the fn input types",
                    br_name,
                )
            };
            if let Some(span) = self.supertrait_span {
                err.span_label(span, "due to this supertrait");
            }
            if !br.is_named(tcx) {
                err.note("lifetimes appearing in an associated or opaque type are not considered constrained");
                err.note("consider introducing a named lifetime parameter");
            }
            err.emit();
        }
    }

    fn output_span(&self, tcx: TyCtxt<'tcx>, region: ty::BoundRegionKind<'tcx>) -> Span {
        // Completed types do not store the HIR span of a nested output. Recover
        // it from the binder's declaration when reporting a deferred error.
        let projection_span = |trait_ref: &hir::TraitRef<'_>| {
            let item = self.associated_item?;
            trait_ref
                .path
                .segments
                .iter()
                .filter_map(|segment| segment.args)
                .flat_map(|args| args.constraints)
                .find(|constraint| constraint.ident.name == tcx.item_name(item))
                .map(|constraint| constraint.span)
        };
        if let ty::BoundRegionKind::Named(def_id) = region
            && let Some(def_id) = def_id.as_local()
        {
            for (_, node) in tcx.hir_parent_iter(tcx.local_def_id_to_hir_id(def_id)) {
                match node {
                    Node::Ty(hir::Ty { kind: hir::TyKind::FnPtr(pointer), .. })
                        if self.associated_item.is_none() =>
                    {
                        return pointer.decl.output.span();
                    }
                    Node::TraitRef(trait_ref) => {
                        return projection_span(trait_ref).unwrap_or(self.span);
                    }
                    Node::Ty(hir::Ty { kind: hir::TyKind::TraitObject(bounds, _), .. }) => {
                        return bounds
                            .iter()
                            .filter(|bound| {
                                bound
                                    .bound_generic_params
                                    .iter()
                                    .any(|param| param.def_id == def_id)
                            })
                            .find_map(|bound| projection_span(&bound.trait_ref))
                            .unwrap_or(self.span);
                    }
                    Node::Item(_) | Node::TraitItem(_) | Node::ImplItem(_) => break,
                    _ => {}
                }
            }
        }
        self.span
    }
}
