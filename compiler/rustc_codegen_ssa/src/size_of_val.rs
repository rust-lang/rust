//! Computing the size and alignment of a value.

use rustc_abi::WrappingRange;
use rustc_hir::LangItem;
use rustc_middle::bug;
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, Ty};
use rustc_span::DUMMY_SP;
use tracing::{debug, trace};

use crate::common::IntPredicate;
use crate::traits::*;
use crate::{common, meth};

pub fn size_and_align_of_dst<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    t: Ty<'tcx>,
    info: Option<Bx::Value>,
) -> (Bx::Value, Bx::Value) {
    let layout = bx.layout_of(t);
    trace!("size_and_align_of_dst(ty={}, info={:?}): layout: {:?}", t, info, layout);
    if layout.is_sized() {
        let size = bx.const_usize(layout.size.bytes());
        let align = bx.const_usize(layout.align.abi.bytes());
        return (size, align);
    }
    match t.kind() {
        ty::Dynamic(..) => {
            // Load size/align from vtable.
            let vtable = info.unwrap();
            let size = meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_SIZE)
                .get_usize(bx, vtable, t);
            let align = meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_ALIGN)
                .get_usize(bx, vtable, t);

            // Size is always <= isize::MAX.
            let size_bound = bx.data_layout().ptr_sized_integer().signed_max() as u128;
            bx.range_metadata(size, WrappingRange { start: 0, end: size_bound });
            // Alignment is always nonzero.
            bx.range_metadata(align, WrappingRange { start: 1, end: !0 });

            (size, align)
        }
        ty::Slice(_) | ty::Str => {
            let unit = layout.field(bx, 0);
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            (
                // All slice sizes must fit into `isize`, so this multiplication cannot
                // wrap -- neither signed nor unsigned.
                bx.unchecked_sumul(info.unwrap(), bx.const_usize(unit.size.bytes())),
                bx.const_usize(unit.align.abi.bytes()),
            )
        }
        ty::Foreign(_) => {
            // `extern` type. We cannot compute the size, so panic.
            let msg_str = with_no_visible_paths!({
                with_no_trimmed_paths!({
                    format!("attempted to compute the size or alignment of extern type `{t}`")
                })
            });
            let msg = bx.const_str(&msg_str);

            // Obtain the panic entry point.
            let (fn_abi, llfn, _instance) =
                common::build_langcall(bx, DUMMY_SP, LangItem::PanicNounwind);

            // Generate the call. Cannot use `do_call` since we don't have a MIR terminator so we
            // can't create a `TerminationCodegenHelper`. (But we are in good company, this code is
            // duplicated plenty of times.)
            let fn_ty = bx.fn_decl_backend_type(fn_abi);

            bx.call(
                fn_ty,
                /* fn_attrs */ None,
                Some(fn_abi),
                llfn,
                &[msg.0, msg.1],
                None,
                None,
            );

            // This function does not return so we can now return whatever we want.
            let size = bx.const_usize(layout.size.bytes());
            let align = bx.const_usize(layout.align.abi.bytes());
            (size, align)
        }
        ty::Adt(..) | ty::Tuple(..) => {
            // First get the size of all statically known fields.
            // Don't use size_of because it also rounds up to alignment, which we
            // want to avoid, as the unsized field's alignment could be smaller.
            assert!(!t.is_simd());
            debug!("DST {} layout: {:?}", t, layout);

            let i = layout.fields.count() - 1;
            let unsized_offset_unadjusted = layout.fields.offset(i).bytes();
            let sized_align = layout.align.abi.bytes();
            debug!(
                "DST {} offset of dyn field: {}, statically sized align: {}",
                t, unsized_offset_unadjusted, sized_align
            );
            let unsized_offset_unadjusted = bx.const_usize(unsized_offset_unadjusted);
            let sized_align = bx.const_usize(sized_align);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let field_ty = layout.field(bx, i).ty;
            let (unsized_size, mut unsized_align) = size_and_align_of_dst(bx, field_ty, info);

            // # First compute the dynamic alignment

            // For packed types, we need to cap the alignment.
            if let ty::Adt(def, _) = t.kind()
                && let Some(packed) = def.repr().pack
            {
                if packed.bytes() == 1 {
                    // We know this will be capped to 1.
                    unsized_align = bx.const_usize(1);
                } else {
                    // We have to dynamically compute `min(unsized_align, packed)`.
                    let packed = bx.const_usize(packed.bytes());
                    let cmp = bx.icmp(IntPredicate::IntULT, unsized_align, packed);
                    unsized_align = bx.select(cmp, unsized_align, packed);
                }
            }

            // Choose max of two known alignments (combined value must
            // be aligned according to more restrictive of the two).
            let full_align = match (
                bx.const_to_opt_u128(sized_align, false),
                bx.const_to_opt_u128(unsized_align, false),
            ) {
                (Some(sized_align), Some(unsized_align)) => {
                    // If both alignments are constant, (the sized_align should always be), then
                    // pick the correct alignment statically.
                    bx.const_usize(std::cmp::max(sized_align, unsized_align) as u64)
                }
                _ => {
                    let cmp = bx.icmp(IntPredicate::IntUGT, sized_align, unsized_align);
                    bx.select(cmp, sized_align, unsized_align)
                }
            };

            // # Then compute the dynamic size

            // The full formula for the size would be:
            // let unsized_offset_adjusted = unsized_offset_unadjusted.align_to(unsized_align);
            // let full_size = (unsized_offset_adjusted + unsized_size).align_to(full_align);
            // However, `unsized_size` is a multiple of `unsized_align`. Therefore, we can
            // equivalently do the `align_to(unsized_align)` *after* adding `unsized_size`:
            //
            // let full_size =
            //     (unsized_offset_unadjusted + unsized_size)
            //     .align_to(unsized_align)
            //     .align_to(full_align);
            //
            // Furthermore, `align >= unsized_align`, and therefore we only need to do:
            // let full_size = (unsized_offset_unadjusted + unsized_size).align_to(full_align);

            let full_size = bx.add(unsized_offset_unadjusted, unsized_size);

            // Issue #27023: must add any necessary padding to `size`
            // (to make it a multiple of `align`) before returning it.
            //
            // Namely, the returned size should be, in C notation:
            //
            //   `size + ((size & (align-1)) ? align : 0)`
            //
            // emulated via the semi-standard fast bit trick:
            //
            //   `(size + (align-1)) & -align`
            let one = bx.const_usize(1);
            let addend = bx.sub(full_align, one);
            let add = bx.add(full_size, addend);
            let neg = bx.neg(full_align);
            let full_size = bx.and(add, neg);

            (full_size, full_align)
        }
        _ => bug!("size_and_align_of_dst: {t} not supported"),
    }
}
