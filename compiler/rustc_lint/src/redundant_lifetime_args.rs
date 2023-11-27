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

    // Compute the implied outlives bounds for the item. This ensures that we treat
    // a signature with an argument like `&'a &'b ()` as implicitly having `'b: 'a`.
    let Ok(assumed_wf_types) = ocx.assumed_wf_types(param_env, owner_id.def_id) else {
        return;
    };
    let implied_bounds = infcx.implied_bounds_tys(param_env, owner_id.def_id, assumed_wf_types);
    let outlives_env = &OutlivesEnvironment::with_bounds(param_env, implied_bounds);

    // The ordering of this lifetime map is a bit subtle.
    //
    // Specifically, we want to find a "candidate" lifetime that precedes a "victim" lifetime,
    // where we can prove that `'candidate = 'victim`.
    //
    // `'static` must come first in this list because we can never replace `'static` with
    // something else, but if we find some lifetime `'a` where `'a = 'static`, we want to
    // suggest replacing `'a` with `'static`.
    let mut lifetimes = vec![tcx.lifetimes.re_static];
    lifetimes.extend(
        ty::GenericArgs::identity_for_item(tcx, owner_id).iter().filter_map(|arg| arg.as_region()),
    );
    // If we are in a function, add its late-bound lifetimes too.
    if matches!(def_kind, DefKind::Fn | DefKind::AssocFn) {
        for var in tcx.fn_sig(owner_id).instantiate_identity().bound_vars() {
            let ty::BoundVariableKind::Region(kind) = var else { continue };
            lifetimes.push(ty::Region::new_late_param(tcx, owner_id.to_def_id(), kind));
        }
    }

    // Keep track of lifetimes which have already been replaced with other lifetimes.
    // This makes sure that if `'a = 'b = 'c`, we don't say `'c` should be replaced by
    // both `'a` and `'b`.
    let mut shadowed = FxHashSet::default();

    for (idx, &candidate) in lifetimes.iter().enumerate() {
        // Don't suggest removing a lifetime twice.
        if shadowed.contains(&candidate) {
            continue;
        }

        // Can't rename a named lifetime named `'_` without ambiguity.
        if !candidate.has_name() {
            continue;
        }

        for &victim in &lifetimes[(idx + 1)..] {
            // We only care about lifetimes that are "real", i.e. that have a def-id.
            let (ty::ReEarlyParam(ty::EarlyParamRegion { def_id, .. })
            | ty::ReLateParam(ty::LateParamRegion {
                bound_region: ty::BoundRegionKind::BrNamed(def_id, _),
                ..
            })) = victim.kind()
            else {
                continue;
            };

            // Do not rename lifetimes not local to this item since they'll overlap
            // with the lint running on the parent. We still want to consider parent
            // lifetimes which make child lifetimes redundant, otherwise we would
            // have truncated the `identity_for_item` args above.
            if tcx.parent(def_id) != owner_id.to_def_id() {
                continue;
            }

            let infcx = infcx.fork();

            // Require that `'candidate = 'victim`
            infcx.sub_regions(SubregionOrigin::RelateRegionParamBound(DUMMY_SP), candidate, victim);
            infcx.sub_regions(SubregionOrigin::RelateRegionParamBound(DUMMY_SP), victim, candidate);

            // If there are no lifetime errors, then we have proven that `'candidate = 'victim`!
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
    /// The lifetime we have found to be redundant.
    victim: ty::Region<'tcx>,
    // The lifetime we can replace the victim with.
    candidate: ty::Region<'tcx>,
}
