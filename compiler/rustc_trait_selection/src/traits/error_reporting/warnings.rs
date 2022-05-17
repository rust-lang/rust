use super::PredicateObligation;

use crate::infer::InferCtxt;

use hir::{ItemKind, TyKind};
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_infer::infer::TraitQueryMode;
use rustc_lint_defs::builtin::DEPRECATED_IN_FUTURE;
use rustc_middle::ty::{self, Binder};

pub trait InferCtxtExt<'tcx> {
    fn check_obligation_lints(
        &self,
        obligation: &PredicateObligation<'tcx>,
        tracer: impl Fn() -> Vec<PredicateObligation<'tcx>>,
    );
}

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    fn check_obligation_lints(
        &self,
        obligation: &PredicateObligation<'tcx>,
        tracer: impl Fn() -> Vec<PredicateObligation<'tcx>>,
    ) {
        if self.query_mode != TraitQueryMode::Standard {
            return;
        }

        if let Some(trait_obligation) = {
            let binder = obligation.predicate.kind();
            match binder.no_bound_vars() {
                None => match binder.skip_binder() {
                    ty::PredicateKind::Trait(data) => Some(obligation.with(binder.rebind(data))),
                    _ => None,
                },
                Some(pred) => match pred {
                    ty::PredicateKind::Trait(data) => Some(obligation.with(Binder::dummy(data))),
                    _ => None,
                },
            }
        } {
            let self_ty = trait_obligation.self_ty().skip_binder();
            let trait_ref = trait_obligation.predicate.skip_binder().trait_ref;

            // FIXME(skippy) lint experiment: deprecate PartialEq on function pointers
            if let ty::FnPtr(_) = self_ty.kind()
                && self.tcx.lang_items().eq_trait() == Some(trait_ref.def_id)
            {
                let node = self.tcx.hir().get(trait_obligation.cause.body_id);
                let mut skip = false;
                if let Node::Item(item) = node
                    && let ItemKind::Impl(impl_item) = &item.kind
                    && let TyKind::BareFn(_) = impl_item.self_ty.kind
                {
                    skip = true;
                }
                if !skip {
                    let root_obligation = tracer().last().unwrap_or(&obligation).clone();
                    self.tcx.struct_span_lint_hir(
                        DEPRECATED_IN_FUTURE,
                        root_obligation.cause.body_id,
                        root_obligation.cause.span,
                        |lint| {
                            lint.build(
                                "FIXME(skippy) PartialEq on function pointers has been deprecated.",
                            )
                            .emit();
                        },
                    );
                }
            }
        }
    }
}
