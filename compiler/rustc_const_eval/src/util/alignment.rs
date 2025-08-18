use rustc_abi::Align;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use tracing::debug;

/// Returns `true` if this place is allowed to be less aligned
/// than its containing struct (because it is within a packed
/// struct).
pub fn is_potentially_misaligned<'tcx, L>(
    tcx: TyCtxt<'tcx>,
    local_decls: &L,
    typing_env: ty::TypingEnv<'tcx>,
    place: Place<'tcx>,
) -> bool
where
    L: HasLocalDecls<'tcx>,
{
    debug!("is_potentially_misaligned({:?})", place);
    let Some(pack) = is_within_packed(tcx, local_decls, place) else {
        debug!("is_potentially_misaligned({:?}) - not within packed", place);
        return false;
    };

    let ty = place.ty(local_decls, tcx).ty;
    let unsized_tail = || tcx.struct_tail_for_codegen(ty, typing_env);

    match tcx.layout_of(typing_env.as_query_input(ty)) {
        Ok(layout) => {
            if layout.align.abi <= pack
                && (layout.is_sized() || matches!(unsized_tail().kind(), ty::Slice(..) | ty::Str))
            {
                // If the packed alignment is greater or equal to the field alignment, the type won't be
                // further disaligned.
                // However we need to ensure the field is sized; for unsized fields, `layout.align` is
                // just an approximation -- except when the unsized tail is a slice, where the alignment
                // is fully determined by the type.
                debug!(
                    "is_potentially_misaligned({:?}) - align = {}, packed = {}; not disaligned",
                    place,
                    layout.align.abi.bytes(),
                    pack.bytes()
                );
                false
            } else {
                true
            }
        }
        Err(_) => {
            // Soundness-critical: this may return false positives (reporting potential misalignment),
            // but must not return false negatives. When layout is unavailable, we stay conservative
            // except for arrays of u8/i8 whose ABI alignment is provably 1.

            // Try to determine alignment from the type structure
            if let Some(element_align) = get_element_alignment(tcx, ty) {
                element_align > pack
            } else {
                // If we still can't determine alignment, conservatively assume disaligned
                true
            }
        }
    }
}

/// Try to determine the alignment of an array element type
fn get_element_alignment<'tcx>(_tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Align> {
    match ty.kind() {
        ty::Array(element_ty, _) | ty::Slice(element_ty) => {
            // Only allow u8 and i8 arrays when layout computation fails
            // Other types are conservatively assumed to be misaligned
            match element_ty.kind() {
                ty::Uint(ty::UintTy::U8) | ty::Int(ty::IntTy::I8) => {
                    // For u8 and i8, we know their alignment is 1
                    Some(Align::from_bytes(1).unwrap())
                }
                _ => {
                    // For other types, we cannot safely determine alignment
                    // Conservatively return None to indicate potential misalignment
                    None
                }
            }
        }
        _ => None,
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
