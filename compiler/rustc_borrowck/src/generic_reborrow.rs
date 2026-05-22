use rustc_hir::Mutability;
use rustc_middle::mir::{Body, BorrowKind, Location, MutBorrowKind, Place};
use rustc_middle::span_bug;
use rustc_middle::ty::{self, AdtDef, GenericArgsRef, Ty, TyCtxt};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum GenericReborrowKind {
    Reborrow,
    CoerceShared,
}

impl GenericReborrowKind {
    pub(crate) fn from_mutability(mutability: Mutability) -> Self {
        match mutability {
            Mutability::Mut => GenericReborrowKind::Reborrow,
            Mutability::Not => GenericReborrowKind::CoerceShared,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            GenericReborrowKind::Reborrow => "Reborrow",
            GenericReborrowKind::CoerceShared => "CoerceShared",
        }
    }

    pub(crate) fn borrow_kind(self) -> BorrowKind {
        match self {
            GenericReborrowKind::Reborrow => BorrowKind::Mut { kind: MutBorrowKind::Default },
            GenericReborrowKind::CoerceShared => BorrowKind::Shared,
        }
    }
}

pub(crate) struct GenericReborrowInfo<'tcx> {
    pub(crate) kind: GenericReborrowKind,
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
    let kind = GenericReborrowKind::from_mutability(mutability);
    let source_ty = source_place.ty(body, tcx).ty;
    let &ty::Adt(source_adt, source_args) = source_ty.kind() else {
        span_bug!(span, "{} source must be an ADT, found `{source_ty}`", kind.name());
    };
    let &ty::Adt(target_adt, target_args) = target_ty.kind() else {
        span_bug!(span, "{} target must be an ADT, found `{target_ty}`", kind.name());
    };

    // The current model supports a single relevant lifetime, carried in the first target
    // argument. Keep that assumption checked here until richer metadata is available.
    let Some(ty::GenericArgKind::Lifetime(target_region)) =
        target_args.get(0).map(|arg| arg.kind())
    else {
        span_bug!(span, "{} target `{target_ty}` does not start with a lifetime", kind.name());
    };

    match kind {
        GenericReborrowKind::Reborrow if source_adt.did() != target_adt.did() => {
            span_bug!(
                span,
                "Reborrow source `{source_ty}` and target `{target_ty}` have different ADTs",
            );
        }
        GenericReborrowKind::CoerceShared if source_adt.did() == target_adt.did() => {
            span_bug!(
                span,
                "CoerceShared source `{source_ty}` and target `{target_ty}` have the same ADT",
            );
        }
        GenericReborrowKind::Reborrow | GenericReborrowKind::CoerceShared => {}
    }

    GenericReborrowInfo {
        kind,
        source_ty,
        source_adt,
        source_args,
        target_adt,
        target_args,
        target_region,
    }
}
