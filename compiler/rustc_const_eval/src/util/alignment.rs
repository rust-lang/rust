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
            // Soundness: For any `T`, the ABI alignment requirement of `[T]` equals that of `T`.
            // Proof sketch:
            //  (1) From `&[T]` we can obtain `&T`, hence align([T]) >= align(T).
            //  (2) Using `std::array::from_ref(&T)` we can obtain `&[T; 1]` (and thus `&[T]`),
            //      hence align(T) >= align([T]).
            // Therefore align([T]) == align(T). Length does not affect alignment.

            // Try to determine alignment from the type structure
            if let Some(element_align) = get_element_alignment(tcx, typing_env, ty) {
                element_align > pack
            } else {
                // If we still can't determine alignment, conservatively assume disaligned
                true
            }
        }
    }
}

/// Returns the ABI alignment of the *element type* if `ty` is an array/slice,
/// otherwise `None`.
///
/// Soundness note:
/// For any `T`, the ABI alignment of `[T]` (and `[T; N]`) equals that of `T`
/// and does not depend on the length `N`.
/// Proof sketch:
///   (1) From `&[T]` we can obtain `&T`  ⇒  align([T]) ≥ align(T).
///   (2) From `&T` we can obtain `&[T; 1]` via `std::array::from_ref`
///       (and thus `&[T]`)  ⇒  align(T) ≥ align([T]).
/// Hence `align([T]) == align(T)`.
///
/// Therefore, when `layout_of([T; N])` is unavailable in generic contexts,
/// it is sufficient (and safe) to use `layout_of(T)` for alignment checks.
///
/// Returns:
/// - `Some(align)` if `ty` is `Array(elem, _)` or `Slice(elem)` and
///    `layout_of(elem)` is available;
/// - `None` otherwise (caller should stay conservative).
fn get_element_alignment<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<Align> {
    match ty.kind() {
        ty::Array(elem_ty, _) | ty::Slice(elem_ty) => {
            // Try to obtain the element's layout; if we can, use its ABI align.
            match tcx.layout_of(typing_env.as_query_input(*elem_ty)) {
                Ok(layout) => Some(layout.align.abi),
                Err(_) => None, // stay conservative when even the element's layout is unknown
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
