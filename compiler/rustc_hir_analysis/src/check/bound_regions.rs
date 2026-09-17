use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_span::Span;
use rustc_trait_selection::traits::bound_regions::output_dependency_param_env;

use crate::hir_ty_lowering::bound_regions::LateBoundRegionCheck;

/// Alias-dependent checks cannot run while `clauses_of` and `fn_sig` are being
/// constructed. Visit their completed, unnormalized values here, including
/// aliases whose other well-formedness requirements are not checked eagerly.
pub(super) fn check_item(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    if !tcx.next_trait_solver_globally() {
        return;
    }
    let kind = tcx.def_kind(def_id);
    if !matches!(
        kind,
        DefKind::Fn
            | DefKind::AssocFn
            | DefKind::Static { .. }
            | DefKind::Const
            | DefKind::AssocConst
            | DefKind::TyAlias
            | DefKind::AssocTy
            | DefKind::OpaqueTy
            | DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Trait
            | DefKind::TraitAlias
            | DefKind::Impl { .. }
    ) {
        return;
    }

    let mut visitor = DependencyVisitor {
        tcx,
        span: tcx.def_span(def_id),
        seen: Default::default(),
        checks: Vec::new(),
    };
    for &(clause, span) in tcx.explicit_clauses_of(def_id).clauses {
        visitor.clause(clause, span);
    }
    if matches!(kind, DefKind::Trait | DefKind::TraitAlias) {
        for &(clause, span) in tcx.explicit_implied_clauses_of(def_id).skip_binder() {
            visitor.clause(clause, span);
        }
    }
    if kind == DefKind::OpaqueTy
        || (kind == DefKind::AssocTy && tcx.is_trait(tcx.parent(def_id.to_def_id())))
    {
        for &(clause, span) in tcx.explicit_item_bounds(def_id).skip_binder() {
            visitor.clause(clause, span);
        }
    }

    for param in &tcx.generics_of(def_id).own_params {
        if let Some(default) = param.default_value(tcx) {
            visitor.span = tcx.def_span(param.def_id);
            default.instantiate_identity().skip_norm_wip().visit_with(&mut visitor);
        }
    }

    match kind {
        DefKind::Fn | DefKind::AssocFn => {
            let signature = tcx.fn_sig(def_id).instantiate_identity().skip_norm_wip();
            let decl = tcx.hir_fn_decl_by_hir_id(tcx.local_def_id_to_hir_id(def_id));
            let span = decl.map_or(tcx.def_span(def_id), |decl| decl.output.span());
            visitor.push(LateBoundRegionCheck::function(tcx, signature, span));
            visitor.span = tcx.def_span(def_id);
            signature.visit_with(&mut visitor);
        }
        DefKind::Struct | DefKind::Union | DefKind::Enum => {
            for field in tcx.adt_def(def_id).all_fields() {
                visitor.span = tcx.ty_span(field.did.expect_local());
                tcx.type_of(field.did)
                    .instantiate_identity()
                    .skip_norm_wip()
                    .visit_with(&mut visitor);
            }
        }
        DefKind::AssocTy
            if tcx.is_trait(tcx.parent(def_id.to_def_id()))
                && !tcx.associated_item(def_id).defaultness(tcx).has_value() => {}
        DefKind::Static { .. }
        | DefKind::Const
        | DefKind::AssocConst
        | DefKind::TyAlias
        | DefKind::AssocTy
        | DefKind::Impl { .. } => {
            visitor.span = tcx.ty_span(def_id);
            tcx.type_of(def_id).instantiate_identity().skip_norm_wip().visit_with(&mut visitor);
        }
        _ => {}
    }

    if !visitor.checks.is_empty() {
        let param_env = output_dependency_param_env(tcx, def_id.to_def_id());
        for check in visitor.checks {
            check.check(tcx, def_id, param_env);
        }
    }
}

struct DependencyVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    span: Span,
    seen: FxHashSet<Ty<'tcx>>,
    checks: Vec<LateBoundRegionCheck<'tcx>>,
}

impl<'tcx> DependencyVisitor<'tcx> {
    fn push(&mut self, check: LateBoundRegionCheck<'tcx>) {
        if check.needs_context(self.tcx) {
            self.checks.push(check);
        }
    }

    fn clause(&mut self, clause: ty::Clause<'tcx>, span: Span) {
        self.span = span;
        if let Some(projection) = clause.as_projection_clause() {
            self.push(LateBoundRegionCheck::projection(projection, span));
        }
        clause.visit_with(self);
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for DependencyVisitor<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if !ty.has_bound_regions() || !self.seen.insert(ty) {
            return;
        }
        match *ty.kind() {
            ty::FnPtr(signature, header) => {
                self.push(LateBoundRegionCheck::function(
                    self.tcx,
                    signature.with(header),
                    self.span,
                ));
            }
            ty::Dynamic(predicates, _) => {
                for predicate in predicates {
                    if let ty::ExistentialPredicate::Projection(projection) =
                        predicate.skip_binder()
                    {
                        let projection = predicate
                            .rebind(projection)
                            .with_self_ty(self.tcx, self.tcx.types.trait_object_dummy_self);
                        self.push(LateBoundRegionCheck::projection(projection, self.span));
                    }
                }
            }
            _ => {}
        }
        ty.super_visit_with(self);
    }
}
