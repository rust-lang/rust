use rustc_hir::Mutability;
use rustc_middle::mir::{Body, Location, Place};
use rustc_middle::span_bug;
use rustc_middle::ty::{self, AdtDef, GenericArgsRef, Ty, TyCtxt};

fn generic_reborrow_name(mutability: Mutability) -> &'static str {
    match mutability {
        Mutability::Mut => "Reborrow",
        Mutability::Not => "CoerceShared",
    }
}

pub(crate) struct GenericReborrowInfo<'tcx> {
    pub(crate) source_ty: Ty<'tcx>,
    pub(crate) source_adt: AdtDef<'tcx>,
    pub(crate) source_args: GenericArgsRef<'tcx>,
    pub(crate) target_adt: AdtDef<'tcx>,
    pub(crate) target_args: GenericArgsRef<'tcx>,
    pub(crate) target_region: ty::Region<'tcx>,
}

pub(crate) fn generic_reborrow_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    location: Location,
    target_ty: Ty<'tcx>,
    mutability: Mutability,
    source_place: Place<'tcx>,
) -> GenericReborrowInfo<'tcx> {
    let span = body.source_info(location).span;
    let name = generic_reborrow_name(mutability);
    let source_ty = source_place.ty(body, tcx).ty;
    let &ty::Adt(source_adt, source_args) = source_ty.kind() else {
        span_bug!(span, "{name} source must be an ADT, found `{source_ty}`");
    };
    let &ty::Adt(target_adt, target_args) = target_ty.kind() else {
        span_bug!(span, "{name} target must be an ADT, found `{target_ty}`");
    };

    // The current model supports a single relevant lifetime, carried in the first target
    // argument. Keep that assumption checked here until richer metadata is available.
    let Some(ty::GenericArgKind::Lifetime(target_region)) =
        target_args.get(0).map(|arg| arg.kind())
    else {
        span_bug!(span, "{name} target `{target_ty}` does not start with a lifetime");
    };

    match mutability {
        Mutability::Mut if source_adt.did() != target_adt.did() => {
            span_bug!(
                span,
                "Reborrow source `{source_ty}` and target `{target_ty}` have different ADTs",
            );
        }
        Mutability::Not if source_adt.did() == target_adt.did() => {
            span_bug!(
                span,
                "CoerceShared source `{source_ty}` and target `{target_ty}` have the same ADT",
            );
        }
        Mutability::Mut | Mutability::Not => {}
    }

    GenericReborrowInfo {
        source_ty,
        source_adt,
        source_args,
        target_adt,
        target_args,
        target_region,
    }
}
