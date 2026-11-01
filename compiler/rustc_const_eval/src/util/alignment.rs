use rustc_abi::Align;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use tracing::debug;

/// If the place may be less aligned than its type requires
/// (because it is in a packed type), returns the AdtDef
/// and packed alignment of its most-unaligned projection.
pub fn place_unalignment<'tcx, L>(
    tcx: TyCtxt<'tcx>,
    local_decls: &L,
    typing_env: ty::TypingEnv<'tcx>,
    place: Place<'tcx>,
) -> Option<(AdtDef<'tcx>, Align)>
where
    L: HasLocalDecls<'tcx>,
{
    let decls = local_decls.local_decls();
    place_unalignment_2(tcx, typing_env, place, |place| place.ty(decls, tcx).ty)
}

/// Like `place_unalignment`, but if you can't supply a `local_decls`.
/// Instead, use `place_ty_fn` to specify a custom method to get the type of a place.
pub fn place_unalignment_2<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    place: Place<'tcx>,
    place_ty_fn: impl Fn(PlaceRef<'tcx>) -> Ty<'tcx>,
) -> Option<(AdtDef<'tcx>, Align)> {
    debug!("unalignment({:?})", place);
    let Some((descr, pack)) = most_packed_projection_2(place, &place_ty_fn) else {
        debug!("unalignment({:?}) - not within packed", place);
        return None;
    };

    let ty = place_ty_fn(place.as_ref());
    let unsized_tail = || tcx.struct_tail_for_codegen(ty, typing_env);
    match tcx.layout_of(typing_env.as_query_input(ty)) {
        Ok(layout)
            if layout.align.abi <= pack
                && (layout.is_sized()
                    || matches!(unsized_tail().kind(), ty::Slice(..) | ty::Str)) =>
        {
            // If the packed alignment is greater or equal to the field alignment, the type won't be
            // further unaligned.
            // However we need to ensure the field is sized; for unsized fields, `layout.align` is
            // just an approximation -- except when the unsized tail is a slice, where the alignment
            // is fully determined by the type.
            debug!(
                "unalignment({:?}) - align = {}, packed = {}; not unaligned",
                place,
                layout.align.bytes(),
                pack.bytes()
            );
            None
        }
        _ => {
            // We cannot figure out the layout. Conservatively assume that this is unaligned.
            debug!("unalignment({:?}) - unaligned", place);
            Some((descr, pack))
        }
    }
}

/// If the place includes a projection from a packed struct,
/// returns the AdtDef and packed alignment of the projection
/// with the lowest pack
pub fn most_packed_projection<'tcx, L>(
    tcx: TyCtxt<'tcx>,
    local_decls: &L,
    place: Place<'tcx>,
) -> Option<(AdtDef<'tcx>, Align)>
where
    L: HasLocalDecls<'tcx>,
{
    let decls = local_decls.local_decls();
    most_packed_projection_2(place, |place| place.ty(decls, tcx).ty)
}

/// Like `most_packed_projection`, but if you can't supply a `local_decls`.
/// Instead, use `place_ty_fn` to specify a custom method to get the type of a place.
pub fn most_packed_projection_2<'tcx>(
    place: Place<'tcx>,
    place_ty_fn: impl Fn(PlaceRef<'tcx>) -> Ty<'tcx>,
) -> Option<(AdtDef<'tcx>, Align)> {
    place
        .iter_projections()
        .rev()
        // Stop at `Deref`; standard ABI alignment applies there.
        .take_while(|(_base, elem)| !matches!(elem, ProjectionElem::Deref))
        // Consider the packed alignments at play here...
        .filter_map(|(base, _elem)| {
            let adt = place_ty_fn(base).ty_adt_def()?;
            let pack = adt.repr().pack?;
            Some((adt, pack))
        })
        // ... and compute their minimum.
        // The overall smallest alignment is what matters.
        .min_by_key(|(_, align)| *align)
}
