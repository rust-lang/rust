//! Computing the size and alignment of a dynamically-sized value.
//!
//! For slice-tailed DSTs (`[T]`, `str`, and arbitrarily nested structs/tuples wrapping them,
//! including `#[repr(align)]` and `#[repr(packed)]`), we distill the entire nesting hierarchy at
//! compile time into an $O(1)$ parameter representation `outer_offset + align_to(base + offset, align)`
//! and emit at most a single multiplication, pre-mask addition, alignment mask (only when
//! `elem_size % align != 0`), and (only when an outer `#[repr(packed)]` struct places a
//! dynamically-padded inner DST at an unaligned offset) post-mask addition.
//!
//! For trait-object-tailed DSTs (`dyn Trait`), we collapse no-op inner wrappers (such as
//! `ManuallyDrop<dyn Trait>` and `UnsafeCell<dyn Trait>`), merge adjacent compatible wrappers, and
//! fold statically capped `#[repr(packed)]` wrappers so that only necessary vtable loads and
//! alignment masks are emitted.

use rustc_abi::{Align, Size, WrappingRange};
use rustc_attr_ir::lang_items::LangItem;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{DUMMY_SP, Span, bug};
use smallvec::SmallVec;
use tracing::{debug, trace};

use crate::common::IntPredicate;
use crate::traits::*;
use crate::{common, meth};

fn codegen_foreign_panic<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(bx: &mut Bx, t: Ty<'tcx>) {
    // `extern` type. We cannot compute the size or alignment, so panic.
    let msg_str = with_no_visible_paths!({
        with_no_trimmed_paths!({
            format!("attempted to compute the size or alignment of extern type `{t}`")
        })
    });
    let msg = bx.const_str(&msg_str);

    // Obtain the panic entry point.
    let (fn_abi, llfn, _instance) = common::build_langcall(bx, DUMMY_SP, LangItem::PanicNounwind);

    // Generate the call. Cannot use `do_call` since we don't have a MIR terminator so we
    // can't create a `TerminationCodegenHelper`. (But we are in good company, this code is
    // duplicated plenty of times.)
    let fn_ty = bx.fn_decl_backend_type(fn_abi);

    bx.call(
        fn_ty,
        /* fn_attrs */ None,
        Some(fn_abi),
        llfn,
        ReturnSlot::Direct, // we know the ABI here
        &[msg.0, msg.1],
        None,
        None,
    );
}

/// Walks from an unsized type `t` down to its unsized tail (`Slice`, `Str`, `Dynamic`, or
/// `Foreign`), returning `(tail_ty, min_pack)` where `min_pack` is the minimum `#[repr(packed)]`
/// alignment cap in bytes encountered along the struct chain (`u64::MAX` if uncapped).
pub fn unsized_tail_and_min_pack<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    mut t: Ty<'tcx>,
) -> (Ty<'tcx>, u64) {
    let mut min_pack = u64::MAX;
    loop {
        match *t.kind() {
            ty::Adt(def, args) if def.is_struct() => {
                if let Some(pack) = def.repr().pack {
                    min_pack = min_pack.min(pack.bytes());
                }
                match def.non_enum_variant().tail_opt() {
                    Some(field) => {
                        t = tcx.normalize_erasing_regions(typing_env, field.ty(tcx, args));
                    }
                    None => break,
                }
            }
            ty::Tuple(tys) if let Some(&last_ty) = tys.last() => {
                t = last_ty;
            }
            ty::Pat(inner, _) => {
                t = inner;
            }
            _ => break,
        }
    }
    (t, min_pack)
}

fn load_vtable_align<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    vtable: Bx::Value,
) -> Bx::Value {
    let align =
        meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_ALIGN).get_usize(bx, vtable);
    // Alignment is always a power of two, thus 1..=0x800…000,
    // but also bounded by the maximum we support in type layout.
    let align_bound = Align::max_for_target(bx.data_layout()).bytes().into();
    bx.range_metadata(align, WrappingRange { start: 1, end: align_bound });
    align
}

/// Computes `max(static_align, min(dyn_align, min_pack))`.
fn clamp_dyn_align<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    mut dyn_align: Bx::Value,
    static_align: u64,
    min_pack: u64,
) -> Bx::Value {
    if min_pack < u64::MAX {
        let packed = bx.const_usize(min_pack);
        let cmp = bx.icmp(IntPredicate::IntULT, dyn_align, packed);
        dyn_align = bx.select(cmp, dyn_align, packed);
    }
    if static_align > 1 {
        let sized_align = bx.const_usize(static_align);
        let cmp = bx.icmp(IntPredicate::IntUGT, sized_align, dyn_align);
        dyn_align = bx.select(cmp, sized_align, dyn_align);
    }
    dyn_align
}

/// If projecting to `field` at static `offset` inside `parent` requires a dynamic alignment
/// adjustment (or panics on an `extern type` tail), emits and returns `Some(dynamic_offset)`.
/// Returns `None` when the field sits at its static `offset` (`0` or `offset.bytes()`).
pub fn dst_field_offset<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    parent: TyAndLayout<'tcx>,
    field: TyAndLayout<'tcx>,
    offset: Size,
    meta: Option<Bx::Value>,
) -> Option<Bx::Value> {
    if field.is_sized() || offset.bytes() == 0 {
        return None;
    }

    // Only the unsized tail field of `parent` can be unsized, so walking from `parent.ty`
    // accumulates `parent`'s `#[repr(packed)]` cap alongside all inner wrapper caps.
    let (tail_ty, min_pack) = unsized_tail_and_min_pack(bx.tcx(), bx.typing_env(), parent.ty);
    match tail_ty.kind() {
        ty::Slice(_) | ty::Str => None,
        ty::Dynamic(..) if offset.bytes().is_multiple_of(min_pack) => None,
        ty::Dynamic(..) => {
            // `offset` is aligned to `min(field.align, parent_pack)`. Since `min_pack <= parent_pack`
            // and `offset` is not a multiple of `min_pack`, we must have `field.align < min_pack <= parent_pack`.
            let vtable_align = load_vtable_align(bx, meta.unwrap());
            let field_static_align = field.align.bytes();
            let offset_bytes = offset.bytes();
            debug_assert!(offset_bytes.is_multiple_of(field_static_align));
            debug_assert!(field_static_align < min_pack);
            if offset_bytes == field_static_align {
                // Since `0 < offset_bytes == field_static_align <= effective_align`, rounding
                // `offset_bytes` up to `effective_align` is simply `effective_align` itself.
                Some(clamp_dyn_align(bx, vtable_align, field_static_align, min_pack))
            } else {
                // `offset_bytes` is already a multiple of `field_static_align`, so all bits below
                // `field_static_align` in `-offset_bytes` are 0. Thus `max(field_static_align, ..)`
                // is unnecessary when computing `offset + (-offset & (align - 1))`.
                let align = clamp_dyn_align(bx, vtable_align, 1, min_pack);
                let align_minus_1 = bx.sub(align, bx.const_usize(1));
                let neg_offset = bx.const_int(bx.type_isize(), offset_bytes.wrapping_neg() as i64);
                let pad = bx.and(neg_offset, align_minus_1);
                Some(bx.add(bx.const_usize(offset_bytes), pad))
            }
        }
        ty::Foreign(_) => {
            codegen_foreign_panic(bx, tail_ty);
            Some(bx.const_usize(offset.bytes()))
        }
        _ => bug!("dst_field_offset: unexpected unsized tail {tail_ty:?}"),
    }
}

/// Computes the alignment of a (possibly unsized) type `t`.
pub fn align_of_dst<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    t: Ty<'tcx>,
    info: Option<Bx::Value>,
    span: Span,
) -> Bx::Value {
    let layout = bx.spanned_layout_of(t, span);
    trace!("align_of_dst(ty={}, info={:?}): layout: {:?}", t, info, layout);
    if layout.is_sized() {
        return bx.const_usize(layout.align.bytes());
    }

    let (tail_ty, min_pack) = unsized_tail_and_min_pack(bx.tcx(), bx.typing_env(), t);
    match tail_ty.kind() {
        ty::Slice(_) | ty::Str => {
            // The static `layout.align.abi` of any slice-tailed DST already incorporates the
            // element alignment as well as all outer fields, `#[repr(align)]`, and `#[repr(packed)]`.
            bx.const_usize(layout.align.bytes())
        }
        ty::Foreign(_) => {
            codegen_foreign_panic(bx, tail_ty);
            bx.const_usize(layout.align.bytes())
        }
        ty::Dynamic(..) => {
            // By distributivity of lattice operations (`max` and `min`) on power-of-two alignments,
            // the alignment of a chain of structs/tuples wrapping `dyn Trait` collapses to:
            //   `max(layout.align.abi, min(vtable.align, min_pack))`.
            // In particular, if `min_pack <= layout.align.abi`, the dynamic term cannot exceed
            // `layout.align.abi`, so the alignment is statically `layout.align.abi`.
            let static_align = layout.align.bytes();
            if min_pack <= static_align {
                return bx.const_usize(static_align);
            }
            let dyn_align = load_vtable_align(bx, info.unwrap());
            clamp_dyn_align(bx, dyn_align, static_align, min_pack)
        }
        _ => bug!("align_of_dst: {t} (tail {tail_ty}) not supported"),
    }
}

#[derive(Copy, Clone)]
struct DstChainStep {
    offset: u64,
    align: u64,
    pack: u64,
}

/// Computes `align_to(base + offset, align)` where `base` is known to be a multiple of
/// `base_multiple` and the final aligned result is guaranteed to fit in `isize`.
fn emit_aligned_static_size<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    base: Bx::Value,
    base_multiple: u64,
    offset: u64,
    align: u64,
) -> Bx::Value {
    if base_multiple.is_multiple_of(align) {
        // Since `base` is a multiple of `align`, `align_to(base + offset, align)` is simply
        // `base + offset.next_multiple_of(align)` with no bitmask needed.
        let addend = offset.next_multiple_of(align);
        return if addend == 0 { base } else { bx.unchecked_suadd(base, bx.const_usize(addend)) };
    }

    // `align_to(base + offset, align) == (base + offset + (align - 1)) & -align`.
    // Because the aligned result is `<= isize::MAX` and `isize::MAX + 1` is a multiple of
    // every valid power-of-two `align`, the unmasked sum `base + offset + (align - 1)` is
    // also `<= isize::MAX` and cannot wrap. We emit `unchecked_uadd` (`add nuw`) rather than
    // `unchecked_suadd` (`add nuw nsw`) before the bitmask: if `nsw` were present and `base`
    // had a known-zero sign bit (e.g. from `mul nuw nsw`), LLVM's `InstCombine` would strip
    // the sign bit from `-align` (turning e.g. `-8` into `0x7fff_ffff_ffff_fff8`), preventing
    // compact sign-extended immediate encoding (`andq $-8`) when `SelectionDAG` cannot
    // re-prove `SignBitIsZero`. `add nuw` still allows LLVM's `ConstantRange` analysis to
    // prove that the masked result is non-zero whenever `offset > 0` (eliminating dead
    // `size == 0` checks in `Box`/`Rc`/`Arc` deallocation).
    let addend = offset + (align - 1);
    let sum = bx.unchecked_uadd(base, bx.const_usize(addend));
    let mask = bx.const_int(bx.type_isize(), align.wrapping_neg() as i64);
    bx.and(sum, mask)
}

fn emit_dyn_step<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    cur_size: Bx::Value,
    base_multiple: Option<u64>,
    step: DstChainStep,
    vtable_align: &mut Option<Bx::Value>,
    vtable: Bx::Value,
) -> Bx::Value {
    if step.pack <= step.align {
        // Alignment at this step is the compile-time constant `step.align`.
        return emit_aligned_static_size(
            bx,
            cur_size,
            base_multiple.unwrap_or(1),
            step.offset,
            step.align,
        );
    }

    let raw_dyn_align = *vtable_align.get_or_insert_with(|| load_vtable_align(bx, vtable));
    let full_align = clamp_dyn_align(bx, raw_dyn_align, step.align, step.pack);

    // At the innermost step (`base_multiple.is_none()`), `cur_size == vtable_size`. When
    // `step.align == 1`, `full_align` divides `vtable_align` and therefore divides `vtable_size`.
    // If `step.offset == 1`, `align_to(1 + vtable_size, full_align)` is simply `vtable_size + full_align`.
    if base_multiple.is_none() && step.offset == 1 && step.align == 1 {
        return bx.unchecked_suadd(cur_size, full_align);
    }

    let sum = if step.offset == 0 {
        let addend = bx.sub(full_align, bx.const_usize(1));
        bx.unchecked_uadd(cur_size, addend)
    } else if step.offset == 1 {
        bx.unchecked_uadd(cur_size, full_align)
    } else {
        let s = bx.unchecked_uadd(cur_size, bx.const_usize(step.offset - 1));
        bx.unchecked_uadd(s, full_align)
    };
    let neg = bx.neg(full_align);
    bx.and(sum, neg)
}

fn emit_dyn_dst_size<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    vtable: Bx::Value,
    steps: &[DstChainStep],
) -> Bx::Value {
    let mut cur_size =
        meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_SIZE).get_usize(bx, vtable);
    // Size is always <= isize::MAX.
    let size_bound = bx.data_layout().ptr_sized_integer().signed_max() as u128;
    bx.range_metadata(cur_size, WrappingRange { start: 0, end: size_bound });

    let mut vtable_align: Option<Bx::Value> = None;
    let mut base_multiple: Option<u64> = None;
    let mut min_pack = u64::MAX;
    let mut pending: Option<DstChainStep> = None;

    // Walk from innermost wrapper (`steps.iter().rev()`) to outermost (`steps[0]`), collapsing
    // no-op and adjacent compatible steps before emitting IR.
    for step in steps.iter().rev() {
        min_pack = min_pack.min(step.pack);
        let cur = DstChainStep { offset: step.offset, align: step.align, pack: min_pack };

        let Some(prev) = &mut pending else {
            // While `pending` is `None`, the current size is `vtable.size`, which is already a
            // multiple of `vtable.align` (and thus also of `min(vtable.align, p)` for any `p`).
            // Any inner wrapper with `offset == 0` and `align == 1` (e.g., `ManuallyDrop`,
            // `UnsafeCell`, `Cell`, `Pin`) is a complete no-op on `vtable.size`.
            if cur.offset == 0 && cur.align == 1 {
                continue;
            }
            pending = Some(cur);
            continue;
        };

        // Whenever `prev.align <= cur.align`, `cur` cannot have `pack < prev.align`, so `cur.offset`
        // is guaranteed by layout calculation to be a multiple of `prev.align`. We can merge `prev`
        // and `cur` into a single step when either:
        // - `cur.offset == 0 && cur.pack == prev.pack` (`cur`'s dynamic alignment is a multiple of
        //   `prev`'s for all `vtable.align`), or
        // - `prev.pack <= prev.align && cur.pack <= cur.align` (both alignments are static constants).
        if prev.align <= cur.align
            && ((cur.offset == 0 && cur.pack == prev.pack)
                || (prev.pack <= prev.align && cur.pack <= cur.align))
        {
            debug_assert!(cur.offset.is_multiple_of(prev.align));
            prev.offset += cur.offset;
            prev.align = cur.align;
            prev.pack = cur.pack;
            continue;
        }

        cur_size = emit_dyn_step(bx, cur_size, base_multiple, *prev, &mut vtable_align, vtable);
        base_multiple = Some(prev.align);
        *prev = cur;
    }

    if let Some(last) = pending {
        cur_size = emit_dyn_step(bx, cur_size, base_multiple, last, &mut vtable_align, vtable);
    }

    cur_size
}

/// Computes the size of a (possibly unsized) type `t`.
pub fn size_of_dst<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    t: Ty<'tcx>,
    info: Option<Bx::Value>,
    span: Span,
) -> Bx::Value {
    let layout = bx.spanned_layout_of(t, span);
    trace!("size_of_dst(ty={}, info={:?}): layout: {:?}", t, info, layout);
    if layout.is_sized() {
        return bx.const_usize(layout.size.bytes());
    }

    let mut steps = SmallVec::<[DstChainStep; 4]>::new();
    let mut cur_layout = layout;
    loop {
        let cur_ty = cur_layout.ty;
        match cur_ty.kind() {
            ty::Adt(..) | ty::Tuple(..) => {
                assert!(!cur_ty.is_simd());
                debug!("DST {} layout: {:?}", cur_ty, cur_layout);

                let i = cur_layout.fields.count() - 1;
                let offset = cur_layout.fields.offset(i).bytes();
                let align = cur_layout.align.bytes();
                let pack = match cur_ty.kind() {
                    ty::Adt(def, _) => def.repr().pack.map_or(u64::MAX, |p| p.bytes()),
                    _ => u64::MAX,
                };
                debug!(
                    "DST {} offset of dyn field: {}, statically sized align: {}, pack: {}",
                    cur_ty, offset, align, pack
                );
                steps.push(DstChainStep { offset, align, pack });
                cur_layout = cur_layout.field(bx, i);
            }
            ty::Pat(..) => {
                cur_layout = cur_layout.field(bx, 0);
            }
            ty::Slice(_) | ty::Str => {
                let unit = cur_layout.field(bx, 0);
                let elem_size = unit.size.bytes();

                if elem_size == 0 {
                    let size = steps
                        .iter()
                        .rev()
                        .fold(0u64, |size, step| (step.offset + size).next_multiple_of(step.align));
                    return bx.const_usize(size);
                }

                let len = info.unwrap();
                let base = if elem_size == 1 {
                    len
                } else {
                    // All slice sizes must fit into `isize`, so this multiplication cannot
                    // wrap -- neither signed nor unsigned.
                    bx.unchecked_sumul(len, bx.const_usize(elem_size))
                };

                // Walk from innermost wrapper (`steps.iter().rev()`) to outermost (`steps[0]`),
                // distilling the chain into `outer_offset + align_to(base + offset, align)`.
                let mut offset = 0u64;
                let mut outer_offset = 0u64;
                let mut align = 1u64;

                for step in steps.iter().rev() {
                    if align <= step.align {
                        offset += (outer_offset + step.offset).next_multiple_of(align);
                        outer_offset = 0;
                        align = step.align;
                        if elem_size.is_multiple_of(align) {
                            offset = offset.next_multiple_of(align);
                            align = 1;
                        }
                    } else {
                        let c = (outer_offset + step.offset).next_multiple_of(step.align);
                        offset += c & !(align - 1);
                        outer_offset = c & (align - 1);
                    }
                }

                let aligned = emit_aligned_static_size(bx, base, elem_size, offset, align);
                return if outer_offset == 0 {
                    aligned
                } else {
                    bx.unchecked_suadd(aligned, bx.const_usize(outer_offset))
                };
            }
            ty::Dynamic(..) => {
                return emit_dyn_dst_size(bx, info.unwrap(), &steps);
            }
            ty::Foreign(_) => {
                codegen_foreign_panic(bx, cur_ty);
                return bx.const_usize(layout.size.bytes());
            }
            _ => bug!("size_of_dst: {t} (tail {cur_ty}) not supported"),
        }
    }
}
