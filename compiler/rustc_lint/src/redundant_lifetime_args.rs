#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{SubregionOrigin, TyCtxtInferExt};
use rustc_macros::LintDiagnostic;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint::builtin::UNUSED_LIFETIMES;
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::{outlives_bounds::InferCtxtExt, ObligationCtxt};

use crate::{LateContext, LateLintPass};

declare_lint_pass!(RedundantLifetimeArgs => []);

impl<'tcx> LateLintPass<'tcx> for RedundantLifetimeArgs {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        check(cx.tcx, cx.param_env, item.owner_id);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        check(cx.tcx, cx.param_env, item.owner_id);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'tcx>) {
        if cx
            .tcx
            .hir()
            .expect_item(cx.tcx.local_parent(item.owner_id.def_id))
            .expect_impl()
            .of_trait
            .is_some()
        {
            // Don't check for redundant lifetimes for trait implementations,
            // since the signature is required to be compatible with the trait.
            return;
        }

        check(cx.tcx, cx.param_env, item.owner_id);
    }
}

fn check<'tcx>(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>, owner_id: hir::OwnerId) {
    let def_kind = tcx.def_kind(owner_id);
    match def_kind {
        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::Fn
        | DefKind::Const
        | DefKind::AssocFn
        | DefKind::AssocConst
        | DefKind::Impl { of_trait: _ } => {
            // Proceed
        }
        DefKind::Mod
        | DefKind::Variant
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Static(_)
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure => return,
    }

    let infcx = &tcx.infer_ctxt().build();
    let ocx = ObligationCtxt::new(infcx);
    let Ok(assumed_wf_types) = ocx.assumed_wf_types(param_env, owner_id.def_id) else {
        return;
    };

    let implied_bounds = infcx.implied_bounds_tys(param_env, owner_id.def_id, assumed_wf_types);
    let outlives_env = &OutlivesEnvironment::with_bounds(param_env, implied_bounds);

    let mut lifetimes = vec![tcx.lifetimes.re_static];
    lifetimes.extend(
        ty::GenericArgs::identity_for_item(tcx, owner_id).iter().filter_map(|arg| arg.as_region()),
    );
    if matches!(def_kind, DefKind::Fn | DefKind::AssocFn) {
        for var in tcx.fn_sig(owner_id).instantiate_identity().bound_vars() {
            let ty::BoundVariableKind::Region(kind) = var else { continue };
            lifetimes.push(ty::Region::new_late_param(tcx, owner_id.to_def_id(), kind));
        }
    }

    // Keep track of lifetimes which have already been replaced with other lifetimes.
    let mut shadowed = FxHashSet::default();

    for (idx, &candidate) in lifetimes.iter().enumerate() {
        if shadowed.contains(&candidate) {
            // Don't suggest removing a lifetime twice.
            continue;
        }

        if !candidate.has_name() {
            // Can't rename a named lifetime with `'_` without ambiguity.
            continue;
        }

        for &victim in &lifetimes[(idx + 1)..] {
            let (ty::ReEarlyParam(ty::EarlyParamRegion { def_id, .. })
            | ty::ReLateParam(ty::LateParamRegion {
                bound_region: ty::BoundRegionKind::BrNamed(def_id, _),
                ..
            })) = victim.kind()
            else {
                continue;
            };

            if tcx.parent(def_id) != owner_id.to_def_id() {
                // Do not rename generics not local to this item since
                // they'll overlap with this lint running on the parent.
                continue;
            }

            let infcx = infcx.fork();
            infcx.sub_regions(SubregionOrigin::RelateRegionParamBound(DUMMY_SP), candidate, victim);
            infcx.sub_regions(SubregionOrigin::RelateRegionParamBound(DUMMY_SP), victim, candidate);
            if infcx.resolve_regions(outlives_env).is_empty() {
                shadowed.insert(victim);
                tcx.emit_spanned_lint(
                    UNUSED_LIFETIMES,
                    tcx.local_def_id_to_hir_id(def_id.expect_local()),
                    tcx.def_span(def_id),
                    RedundantLifetimeArgsLint { candidate, victim },
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_redundant_lifetime_args)]
#[note]
struct RedundantLifetimeArgsLint<'tcx> {
    candidate: ty::Region<'tcx>,
    victim: ty::Region<'tcx>,
}
