use std::ops::ControlFlow;

use rustc_abi::{Layout, LayoutData};
use rustc_attr_ir::lang_items::LangItem;
use rustc_middle::queries::Providers;
use rustc_middle::ty::{
    self, Instance, InstanceKind, ParamTy, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitor,
};
use tracing::{debug, instrument};

// TODO: this should be used less ad hoc
#[instrument(level = "debug", skip(tcx), ret)]
fn polymorphize_instance<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Instance<'tcx> {
    if !tcx.sess.opts.unstable_opts.polymorphize {
        return instance;
    }
    let InstanceKind::Item(def_id) = instance.def else {
        return instance;
    };
    if tcx.intrinsic(def_id).is_some() {
        // Intrinsics sometimes do weird stuff with generics, where they are generic
        // but rustc will ICE if they are called with unexpected types.
        // TODO: is there any way this is visible from generic user code that doesn't use intrinsics?
        return instance;
    }
    // FIXME: exclude all functions with non-Rust reprs anywhere in signature (including theirs)

    let generics = tcx.generics_of(def_id);
    let clauses: Vec<_> = tcx
        .clauses_of(def_id)
        .instantiate_identity(tcx)
        .into_iter()
        .map(|(c, _)| c.skip_norm_wip())
        .collect();
    let args =
        tcx.mk_args_from_iter(instance.args.iter().enumerate().map(
            |(idx, arg)| match arg.kind() {
                ty::GenericArgKind::Type(ty) => {
                    let param_ty = ParamTy::for_def(generics.param_at(idx, tcx));
                    let layout = tcx
                        .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty))
                        .unwrap()
                        .layout;
                    if is_ty_param_polymorphizable(tcx, &clauses, param_ty)
                        // FIXME: pass drop glue in sidecar to relax this
                        && !ty.needs_drop(tcx, ty::TypingEnv::fully_monomorphized())
                        // FIXME: track metadata layout in ParamLayout to relax this
                        && layout.is_sized()
                        // FIXME: figure out how to handle uninhabited types -- should they be erased?
                        // (causes issues with comparing layout inhabitedness w/ opsem inhabitedness)
                        && !layout.uninhabited
                    {
                        let param_layout = make_param_layout(tcx, layout);
                        let erased_ty =
                            tcx.mk_ty_from_kind(ty::Erased(Some(param_ty), param_layout));
                        erased_ty.into()
                    } else {
                        arg
                    }
                }
                ty::GenericArgKind::Lifetime(_) | ty::GenericArgKind::Const(_) => arg,
            },
        ));
    Instance { def: instance.def, args }
}

fn make_param_layout<'tcx>(tcx: TyCtxt<'tcx>, layout: Layout<'tcx>) -> ty::ParamLayout<'tcx> {
    let LayoutData {
        fields: _,
        variants: _,
        backend_repr,
        largest_niche,
        uninhabited: _,
        align: rustc_abi::AbiAlign { abi: align },
        size,
        max_repr_align: _,
        unadjusted_abi_align: _,
        randomization_seed: _,
    } = *layout.0.0;
    tcx.mk_param_layout(ty::ParamLayoutData { backend_repr, largest_niche, align, size })
}

#[instrument(level = "debug", ret, skip(tcx))]
fn is_ty_param_polymorphizable<'tcx>(
    tcx: TyCtxt<'tcx>,
    clauses: &[ty::Clause<'tcx>],
    param_ty: ParamTy,
) -> bool {
    for clause in clauses {
        // FIXME: is skip_binder appropriate?
        match clause.kind().skip_binder() {
        ty::ClauseKind::Trait(trait_predicate) => {
            let is_for_param = trait_predicate.self_ty().is_param(param_ty.index);
            if is_for_param
                && tcx.is_lang_item(trait_predicate.def_id(), LangItem::Sized)
            {
                // FIXME: what about rest of Sized hierarchy?
                debug!("found Sized bound -> doesn't matter");
            } else if has_param(param_ty, trait_predicate) {
                debug!("polymorphization disqualified by {clause:?}");
                return false;
            }
        }
        ty::ClauseKind::Projection(projection_predicate) => {
            if has_param(param_ty, projection_predicate) {
                debug!("polymorphization disqualified by {clause:?}");
                return false;
            }
        }
        ty::ClauseKind::ConstArgHasType(_, ty) => {
            if has_param(param_ty, ty) {
                debug!("polymorphization disqualified by {clause:?}");
                return false;
            }
        }
        ty::ClauseKind::RegionOutlives(..)
        // TODO: maybe disqualify T: 'static since that's needed for TypeId on stable?
        // might allow ignoring TypeId for now without breaking stuff.
        | ty::ClauseKind::TypeOutlives(..)
        | ty::ClauseKind::WellFormed(..)
        | ty::ClauseKind::HostEffect(..)
        | ty::ClauseKind::ConstEvaluatable(..)
        | ty::ClauseKind::UnstableFeature(..) => {}
    }
    }
    true
}

fn has_param<'tcx, T: TypeVisitable<TyCtxt<'tcx>>>(param_ty: ParamTy, t: T) -> bool {
    t.visit_with(&mut HasParamVisitor { param_ty }).is_break()
}

struct HasParamVisitor {
    param_ty: ParamTy,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for HasParamVisitor {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        if t.is_param(self.param_ty.index) {
            return ControlFlow::Break(());
        }
        t.super_visit_with(self)
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.polymorphize_instance = polymorphize_instance;
}
