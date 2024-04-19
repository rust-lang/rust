use rustc_data_structures::{fx::FxIndexSet, unord::UnordSet};
use rustc_errors::LintDiagnostic;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_span::Span;

use crate::fluent_generated as fluent;
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// UwU
    pub IMPL_TRAIT_OVERCAPTURES,
    Warn,
    "will capture more lifetimes than possibly intended in edition 2024",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/IntoIterator-for-arrays.html>",
    };
}

declare_lint_pass!(
    /// Lint for use of `async fn` in the definition of a publicly-reachable
    /// trait.
    ImplTraitOvercaptures => [IMPL_TRAIT_OVERCAPTURES]
);

impl<'tcx> LateLintPass<'tcx> for ImplTraitOvercaptures {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        _: &'tcx hir::Body<'tcx>,
        _: Span,
        parent_def_id: LocalDefId,
    ) {
        match cx.tcx.def_kind(parent_def_id) {
            DefKind::AssocFn => {
                // RPITITs already capture all lifetimes in scope, so skip them.
                if matches!(
                    cx.tcx.def_kind(cx.tcx.local_parent(parent_def_id)),
                    DefKind::Trait | DefKind::Impl { of_trait: true }
                ) {
                    return;
                }
            }
            DefKind::Fn => {
                // All freee functions need to check for overcaptures.
            }
            DefKind::Closure => return,
            kind => {
                unreachable!(
                    "expected function item, found {}",
                    kind.descr(parent_def_id.to_def_id())
                )
            }
        }

        let sig = cx.tcx.fn_sig(parent_def_id).instantiate_identity();

        let mut in_scope_parameters = FxIndexSet::default();
        let mut current_def_id = Some(parent_def_id.to_def_id());
        while let Some(def_id) = current_def_id {
            let generics = cx.tcx.generics_of(def_id);
            for param in &generics.params {
                in_scope_parameters.insert(param.def_id);
            }
            current_def_id = generics.parent;
        }

        sig.visit_with(&mut VisitOpaqueTypes {
            tcx: cx.tcx,
            parent_def_id,
            in_scope_parameters,
            seen: Default::default(),
        });
    }
}

struct VisitOpaqueTypes<'tcx> {
    tcx: TyCtxt<'tcx>,
    parent_def_id: LocalDefId,
    in_scope_parameters: FxIndexSet<DefId>,
    seen: FxIndexSet<LocalDefId>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for VisitOpaqueTypes<'tcx> {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        t: &ty::Binder<'tcx, T>,
    ) -> Self::Result {
        let mut added = vec![];
        for arg in t.bound_vars() {
            let arg: ty::BoundVariableKind = arg;
            match arg {
                ty::BoundVariableKind::Region(ty::BoundRegionKind::BrNamed(def_id, ..)) => {
                    added.push(def_id);
                    let unique = self.in_scope_parameters.insert(def_id);
                    assert!(unique);
                }
                ty::BoundVariableKind::Ty(_) => {
                    todo!("we don't support late-bound type params in `impl Trait`")
                }
                ty::BoundVariableKind::Region(..) => {
                    unreachable!("all AST-derived bound regions should have a name")
                }
                ty::BoundVariableKind::Const => {
                    unreachable!("non-lifetime binder consts are not allowed")
                }
            }
        }

        t.super_visit_with(self);

        for arg in added.into_iter().rev() {
            self.in_scope_parameters.shift_remove(&arg);
        }
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        if !t.has_opaque_types() {
            return;
        }

        if let ty::Alias(ty::Opaque, opaque_ty) = *t.kind()
            && let Some(opaque_def_id) = opaque_ty.def_id.as_local()
            && self.seen.insert(opaque_def_id)
            && let opaque =
                self.tcx.hir_node_by_def_id(opaque_def_id).expect_item().expect_opaque_ty()
            && let hir::OpaqueTyOrigin::FnReturn(parent_def_id) = opaque.origin
            && parent_def_id == self.parent_def_id
            && opaque.precise_capturing_args.is_none()
        {
            let mut captured = UnordSet::default();
            let variances = self.tcx.variances_of(opaque_def_id);
            let mut current_def_id = Some(opaque_def_id.to_def_id());
            while let Some(def_id) = current_def_id {
                let generics = self.tcx.generics_of(def_id);
                for param in &generics.params {
                    if variances[param.index as usize] != ty::Invariant {
                        continue;
                    }
                    captured.insert(extract_def_id_from_arg(
                        self.tcx,
                        generics,
                        opaque_ty.args[param.index as usize],
                    ));
                }
                current_def_id = generics.parent;
            }

            let uncaptured_spans: Vec<_> = self
                .in_scope_parameters
                .iter()
                .filter(|def_id| !captured.contains(def_id))
                .map(|def_id| self.tcx.def_span(def_id))
                .collect();

            if !uncaptured_spans.is_empty() {
                self.tcx.emit_node_lint(
                    IMPL_TRAIT_OVERCAPTURES,
                    self.tcx.local_def_id_to_hir_id(opaque_def_id),
                    ImplTraitOvercapturesLint {
                        opaque_span: self.tcx.def_span(opaque_def_id),
                        self_ty: t,
                        num_captured: uncaptured_spans.len(),
                        uncaptured_spans,
                    },
                );
            }

            for clause in
                self.tcx.item_bounds(opaque_ty.def_id).iter_instantiated(self.tcx, opaque_ty.args)
            {
                clause.visit_with(self)
            }
        }

        t.super_visit_with(self);
    }
}

struct ImplTraitOvercapturesLint<'tcx> {
    opaque_span: Span,
    uncaptured_spans: Vec<Span>,
    self_ty: Ty<'tcx>,
    num_captured: usize,
}

impl<'a> LintDiagnostic<'a, ()> for ImplTraitOvercapturesLint<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.arg("self_ty", self.self_ty.to_string())
            .arg("num_captured", self.num_captured)
            .span(self.opaque_span)
            .span_note(self.uncaptured_spans, fluent::lint_note)
            .note(fluent::lint_note2);
    }

    fn msg(&self) -> rustc_errors::DiagMessage {
        fluent::lint_impl_trait_overcaptures
    }
}

fn extract_def_id_from_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: &'tcx ty::Generics,
    arg: ty::GenericArg<'tcx>,
) -> DefId {
    match arg.unpack() {
        ty::GenericArgKind::Lifetime(re) => match *re {
            ty::ReEarlyParam(ebr) => generics.region_param(ebr, tcx).def_id,
            ty::ReBound(
                _,
                ty::BoundRegion { kind: ty::BoundRegionKind::BrNamed(def_id, ..), .. },
            ) => def_id,
            _ => unreachable!(),
        },
        ty::GenericArgKind::Type(ty) => {
            let ty::Param(param_ty) = *ty.kind() else {
                bug!();
            };
            generics.type_param(param_ty, tcx).def_id
        }
        ty::GenericArgKind::Const(ct) => {
            let ty::ConstKind::Param(param_ct) = ct.kind() else {
                bug!();
            };
            generics.const_param(param_ct, tcx).def_id
        }
    }
}
