use rustc_abi::Align;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use tracing::debug;

/// Returns `true` if this place is allowed to be less aligned
/// than its containing struct (because it is within a packed
/// struct).
pub fn is_disaligned<'tcx, L>(
    tcx: TyCtxt<'tcx>,
    local_decls: &L,
    typing_env: ty::TypingEnv<'tcx>,
    place: Place<'tcx>,
) -> bool
where
    L: HasLocalDecls<'tcx>,
{
    debug!("is_disaligned({:?})", place);
    let Some(pack) = is_within_packed(tcx, local_decls, place) else {
        debug!("is_disaligned({:?}) - not within packed", place);
        return false;
    };

    let ty = place.ty(local_decls, tcx).ty;
    let unsized_tail = || tcx.struct_tail_for_codegen(ty, typing_env);
    match tcx.layout_of(typing_env.as_query_input(ty)) {
        Ok(layout)
            if layout.align.abi <= pack
                && (layout.is_sized()
                    || matches!(unsized_tail().kind(), ty::Slice(..) | ty::Str)) =>
        {
            // If the packed alignment is greater or equal to the field alignment, the type won't be
            // further disaligned.
            // However we need to ensure the field is sized; for unsized fields, `layout.align` is
            // just an approximation -- except when the unsized tail is a slice, where the alignment
            // is fully determined by the type.
            debug!(
                "is_disaligned({:?}) - align = {}, packed = {}; not disaligned",
                place,
                layout.align.bytes(),
                pack.bytes()
            );
            false
        }
        _ => {
            // We cannot figure out the layout. Conservatively assume that this is disaligned.
            debug!("is_disaligned({:?}) - true", place);
            true
        }
    }
}

pub fn is_within_packed<'tcx, L>(
    tcx: TyCtxt<'tcx>,
    local_decls: &L,
    place: Place<'tcx>,
) -> Option<Align>
where
    L: HasLocalDecls<'tcx>,
{
    place
        .iter_projections()
        .rev()
        // Stop at `Deref`; standard ABI alignment applies there.
        .take_while(|(_base, elem)| !matches!(elem, ProjectionElem::Deref))
        // Consider the packed alignments at play here...
        .filter_map(|(base, _elem)| {
            base.ty(local_decls, tcx).ty.ty_adt_def().and_then(|adt| adt.repr().pack)
        })
        // ... and compute their minimum.
        // The overall smallest alignment is what matters.
        .min()
}
