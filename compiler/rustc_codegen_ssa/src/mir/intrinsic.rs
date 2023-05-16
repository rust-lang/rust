use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::FunctionCx;
use crate::common::IntPredicate;
use crate::errors;
use crate::errors::InvalidMonomorphization;
use crate::glue;
use crate::meth;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::ty;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::sym;
use rustc_span::Span;
use rustc_target::abi::call::{FnAbi, PassMode};
use rustc_target::abi::WrappingRange;

fn copy_intrinsic<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    allow_overlap: bool,
    volatile: bool,
    ty: Ty<'tcx>,
    dst: Bx::Value,
    src: Bx::Value,
    count: Bx::Value,
) {
    let layout = bx.layout_of(ty);
    let size = layout.size;
    let align = layout.align.abi;
    let size = bx.mul(bx.const_usize(size.bytes()), count);
    let flags = if volatile { MemFlags::VOLATILE } else { MemFlags::empty() };
    if allow_overlap {
        bx.memmove(dst, align, src, align, size, flags);
    } else {
        bx.memcpy(dst, align, src, align, size, flags);
    }
}

mod swap_intrinsic {
    use crate::traits::*;
    use crate::MemFlags;

    use rustc_middle::mir::interpret::PointerArithmetic;
    use rustc_middle::ty::Ty;
    use rustc_span::Span;
    use rustc_target::abi::{Align, Size};
    use rustc_target::spec::HasTargetSpec;

    // Note: We deliberately interpret our values as some ranges of bytes
    // for performance like did earlier in the old `core::mem::swap` implementation
    // and use immediate values instead of PlaceRefs.
    pub(super) fn single<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        x_ptr: Bx::Value,
        y_ptr: Bx::Value,
        ty: Ty<'tcx>,
        span: Span,
    ) {
        let layout = bx.layout_of(ty);
        if layout.is_unsized() {
            span_bug!(span, "swap_nonoverlapping_single must be called only for sized types");
        }
        if layout.is_zst() {
            // no-op
            return;
        }
        let should_use_2_temp_vals = {
            // Primitive integer or something equal to it by size.
            (layout.size <= bx.cx().pointer_size() && layout.size.bytes().is_power_of_two())
            // SPIR-V doesn't allow reinterpretation of values as chunks of arbitrary ints
            // so we need to read and copy them full.
            // For small values we use double read-double write.
            || (layout.size <= bx.cx().pointer_size() && bx.cx().target_spec().arch == "spirv")
        };
        if should_use_2_temp_vals {
            let ty = bx.backend_type(layout);
            let align = layout.align.abi;
            swap_using_2_temps(bx, x_ptr, y_ptr, ty, align);
            return;
        }

        // If need to swap large value,
        // it probably better to do single memcpy from one elem
        // to another after saving the old value.
        let should_use_single_temp_val = {
            // Most likely some `Simd<X, N>` type from portable simd or manual simd.
            // There is no difference with double read in release build
            // but it reduces amount of code generated in debug build.
            (layout.align.abi.bytes() == layout.size.bytes() && layout.size > bx.cx().pointer_size())
            // Probably aggregate with some SIMD type field.
            // E.g. `Option<f32x4>`.
            // Need to think how to do it better.
            || layout.align.abi > bx.data_layout().pointer_align.abi
            // SPIRV doesn't allow partial reads/writes and value reinterpretations
            // so our best chance to reduce stack usage is to use single alloca.
            || bx.cx().target_spec().arch == "spirv"
        };
        if should_use_single_temp_val {
            let ty = bx.backend_type(layout);
            swap_using_single_temp(bx, x_ptr, y_ptr, ty, layout.size, layout.align.abi);
            return;
        }

        // Both LLVM and GCC seem to benefit from same splitting loops
        // so place this code here to prevent duplication.
        // https://godbolt.org/z/arzvePb8T

        if bx.cx().target_spec().arch == "x86_64" {
            swap_unaligned_x86_64_single(bx, layout, x_ptr, y_ptr);
            return;
        }

        // Swap using aligned integers as chunks.
        assert!(layout.align.abi.bytes() <= bx.pointer_size().bytes());
        assert_eq!(bx.data_layout().pointer_align.abi.bytes(), bx.pointer_size().bytes());
        let chunk_size = std::cmp::min(layout.align.abi.bytes(), bx.pointer_size().bytes());
        let chunk_size = Size::from_bytes(chunk_size);
        make_swaps_loop(
            bx,
            x_ptr,
            y_ptr,
            ToSwap::Bytes(layout.size),
            ChunkInfo::IntChunk(chunk_size),
            NumOfTemps::Two,
            Align::from_bytes(chunk_size.bytes()).unwrap(),
        );
    }

    // `x86_64` allows optimization using unaligned accesses
    // because unaligned reads/writes are fast on x86_64.
    // https://lemire.me/blog/2012/05/31/data-alignment-for-speed-myth-or-reality/
    // We manually swap last `x % ZMM_BYTES` bytes in a way that would always vectorize
    // them AVX and/or SSE because both GCC and LLVM generate fails to use smaller SIMD registers
    // if they had used larger ones.
    fn swap_unaligned_x86_64_single<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        layout: Bx::LayoutOfResult,
        x_ptr: Bx::Value,
        y_ptr: Bx::Value,
    ) {
        const ZMM_BYTES: u64 = 512 / 8;
        const YMM_BYTES: u64 = 256 / 8;
        const XMM_BYTES: u64 = 128 / 8;

        let min_align = Align::from_bytes(1).expect("One is always valid align.");
        let ptr_size = bx.cx().pointer_size();
        // Need to do pointercasts because `rustc_codegen_gcc` ignores passed type
        // in `inbounds_gep`.
        let x_ptr = bx.pointercast(x_ptr, bx.type_i8p());
        let y_ptr = bx.pointercast(y_ptr, bx.type_i8p());

        let mut total_offset = Size::ZERO;
        // Make a loop that is vectorized using largest vectors.
        // It would use largest available vectors, not necessary ZMM.
        if layout.size.bytes() >= ZMM_BYTES {
            let to_swap = Size::from_bytes(layout.size.bytes() / ZMM_BYTES * ZMM_BYTES);
            make_swaps_loop(
                bx,
                x_ptr,
                y_ptr,
                ToSwap::Bytes(to_swap),
                ChunkInfo::IntChunk(ptr_size),
                NumOfTemps::Two,
                min_align,
            );
            total_offset += to_swap;
        }
        // This loop contents are based on knowledge from this: https://godbolt.org/z/Mr4rWfoad
        // And this: https://godbolt.org/z/YzcWofG5Y
        // Both LLVM and GCC fail to use SIMD registers for swapping tails without this.
        for (num_temps, chunk_size) in [(4, YMM_BYTES), (2, XMM_BYTES)] {
            let chunk_size = Size::from_bytes(chunk_size);
            assert_eq!(
                ptr_size * num_temps,
                chunk_size,
                "Invalid assumption about pointer size or register size",
            );
            if layout.size < total_offset + chunk_size {
                continue;
            }

            let x_tmps_and_offsets: Vec<_> = (0..num_temps)
                .map(|i| {
                    let curr_off = total_offset + i * ptr_size;
                    let curr_off = bx.const_usize(curr_off.bytes());
                    let x_gep = bx.inbounds_gep(bx.type_i8(), x_ptr, &[curr_off]);
                    // FIXME: Remove pointercast after stopping support of LLVM 14.
                    let x_gep = bx.pointercast(x_gep, bx.type_ptr_to(bx.type_isize()));
                    (bx.load(bx.type_isize(), x_gep, min_align), curr_off)
                })
                .collect();

            let chunk_size_val = bx.const_usize(chunk_size.bytes());
            let chunk_offset = bx.const_usize(total_offset.bytes());
            let x_chunk_gep = bx.inbounds_gep(bx.type_i8(), x_ptr, &[chunk_offset]);
            let y_chunk_gep = bx.inbounds_gep(bx.type_i8(), y_ptr, &[chunk_offset]);
            // FIXME(AngelicosPhosphoros): Use memcpy.inline here.
            bx.memcpy(
                x_chunk_gep,
                min_align,
                y_chunk_gep,
                min_align,
                chunk_size_val,
                MemFlags::UNALIGNED,
            );
            for (x_tmp, curr_off) in x_tmps_and_offsets {
                let y_gep = bx.inbounds_gep(bx.type_i8(), y_ptr, &[curr_off]);
                // FIXME: Remove pointercast after stopping support of LLVM 14.
                let y_gep = bx.pointercast(y_gep, bx.type_ptr_to(bx.type_isize()));
                bx.store(x_tmp, y_gep, min_align);
            }

            total_offset += chunk_size;
        }

        // I decided to use swaps by pow2 ints here based
        // on this codegen example: https://godbolt.org/z/rWYqMGnWh
        // This loops implements it using minimal amount of instructions
        // and registers involved.
        let mut current_size = bx.pointer_size();
        while total_offset < layout.size {
            // In each loop iteration, remaining amount of unswapped bytes
            // is less than in previous iteration.

            assert_ne!(current_size, Size::ZERO, "We must had finished swapping when it was 1");

            let next_size = Size::from_bytes(current_size.bytes() / 2);
            if total_offset + current_size > layout.size {
                current_size = next_size;
                continue;
            }

            let tail_offset = bx.const_usize(total_offset.bytes());
            let x_tail_ptr = bx.inbounds_gep(bx.type_i8(), x_ptr, &[tail_offset]);
            let y_tail_ptr = bx.inbounds_gep(bx.type_i8(), y_ptr, &[tail_offset]);

            let chunt_ty = choose_int_by_size(bx, current_size);
            swap_using_2_temps(bx, x_tail_ptr, y_tail_ptr, chunt_ty, min_align);

            total_offset += current_size;
            current_size = next_size;
        }
    }

    // We cannot use some of optimizations available for [`single`]
    // because we don't know how many bytes exactly we need to swap.
    pub(super) fn many<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        x_ptr: Bx::Value,
        y_ptr: Bx::Value,
        count: Bx::Value,
        ty: Ty<'tcx>,
        span: Span,
    ) {
        let layout = bx.layout_of(ty);
        if layout.is_unsized() {
            span_bug!(span, "swap_nonoverlapping_many must be called only for sized types");
        }
        if layout.is_zst() {
            // no-op
            return;
        }

        let must_not_split_values = {
            // Unusual type, maybe some manual SIMD optimization.
            layout.align.abi > bx.data_layout().pointer_align.abi && layout.align.abi.bytes() == layout.size.bytes()
            // Probably aggregate with some SIMD type field.
            // E.g. `Option<f32x4>`.
            // Need to think how to do it better.
            || layout.align.abi > bx.data_layout().pointer_align.abi
            // SPIR-V doesn't allow reinterpretation of values as chunks of arbitrary ints
            // so we need to read and copy them by element full.
            || bx.cx().target_spec().arch == "spirv"
        };

        if must_not_split_values {
            let back_ty = bx.backend_type(layout);
            let num_of_temps =
                if layout.size > bx.pointer_size() { NumOfTemps::Single } else { NumOfTemps::Two };
            make_swaps_loop(
                bx,
                x_ptr,
                y_ptr,
                ToSwap::Iterations(count),
                ChunkInfo::RealTyChunk(back_ty, layout.size),
                num_of_temps,
                layout.align.abi,
            );
            return;
        }

        let chunk_size = if bx.cx().target_spec().arch == "x86_64" {
            // x86_64 allows unaligned reads/writes
            // and it is relatively fast
            // so try largest chunk available.
            const INT_SIZES: [u64; 4] = [1, 2, 4, 8];
            INT_SIZES
                .into_iter()
                .map(Size::from_bytes)
                .take_while(|x| *x <= layout.size)
                .filter(|x| layout.size.bytes() % x.bytes() == 0)
                .last()
                .unwrap()
        } else {
            // Fallback to integer with size equal to alignment
            Size::from_bytes(layout.align.abi.bytes())
        };

        let chunks_per_elem = layout.size.bytes() / chunk_size.bytes();
        assert_ne!(chunks_per_elem, 0);
        let iterations = if chunks_per_elem == 1 {
            count
        } else {
            let chunks_per_elem = bx.const_usize(chunks_per_elem);
            bx.unchecked_umul(count, chunks_per_elem)
        };

        make_swaps_loop(
            bx,
            x_ptr,
            y_ptr,
            ToSwap::Iterations(iterations),
            ChunkInfo::IntChunk(chunk_size),
            NumOfTemps::Two,
            // It iterates either by chunks equal to alignment
            // or multiply of alignment so it would always be correct.
            layout.align.abi,
        );
    }

    fn choose_int_by_size<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        size: Size,
    ) -> Bx::Type {
        match size.bits() {
            8 => bx.type_i8(),
            16 => bx.type_i16(),
            32 => bx.type_i32(),
            64 => bx.type_i64(),
            128 => bx.type_i128(),
            _ => unreachable!("Unexpected target int {:?}.", size),
        }
    }

    #[derive(Clone, Copy)]
    enum ToSwap<BxValue> {
        /// Size of region to swap. Useful when we know exact value.
        Bytes(Size),
        /// Number of chunks to swap. For runtime value.
        Iterations(BxValue),
    }

    #[derive(Clone, Copy)]
    enum ChunkInfo<BxType> {
        /// When we want to use it directly
        RealTyChunk(BxType, Size),
        /// When we want to split value by integer chunk.
        IntChunk(Size),
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    enum NumOfTemps {
        Single,
        Two,
    }

    fn make_swaps_loop<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        x_ptr: Bx::Value,
        y_ptr: Bx::Value,
        to_swap: ToSwap<Bx::Value>,
        chunk_info: ChunkInfo<Bx::Type>,
        num_of_temps: NumOfTemps,
        access_align: Align,
    ) {
        let (ChunkInfo::IntChunk(chunk_size) | ChunkInfo::RealTyChunk(_, chunk_size)) = chunk_info;

        assert_ne!(chunk_size, Size::ZERO);

        if let ToSwap::Bytes(total_bytes) = to_swap {
            assert!(
                total_bytes > chunk_size,
                "No need to generate loop when simple swap is enough."
            );
            assert_eq!(
                total_bytes.bytes() % chunk_size.bytes(),
                0,
                "Cannot split size of swap into chunks."
            );
        }

        assert_eq!(
            chunk_size.bytes() % access_align.bytes(),
            0,
            "Ensure that access align doesn't shift",
        );

        let chunk_ty = match chunk_info {
            ChunkInfo::RealTyChunk(ty, _) => ty,
            ChunkInfo::IntChunk(size) => choose_int_by_size(bx, size),
        };

        let iterations = match to_swap {
            ToSwap::Bytes(s) => {
                let iterations_val = s.bytes() / chunk_size.bytes();
                bx.const_usize(iterations_val)
            }
            ToSwap::Iterations(it) => it,
        };

        // Need to do pointercasts because `rustc_codegen_gcc` ignores passed type
        // in `inbounds_gep`.
        let x_ptr = bx.pointercast(x_ptr, bx.type_i8p());
        let y_ptr = bx.pointercast(y_ptr, bx.type_i8p());
        bx.make_memory_loop(
            "swap_loop",
            [x_ptr, y_ptr],
            [chunk_size; 2],
            iterations,
            |body_bx, &[curr_x_ptr, curr_y_ptr]| match num_of_temps {
                NumOfTemps::Single => swap_using_single_temp(
                    body_bx,
                    curr_x_ptr,
                    curr_y_ptr,
                    chunk_ty,
                    chunk_size,
                    access_align,
                ),
                NumOfTemps::Two => {
                    swap_using_2_temps(body_bx, curr_x_ptr, curr_y_ptr, chunk_ty, access_align)
                }
            },
        );
    }

    fn swap_using_2_temps<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        x_ptr: Bx::Value,
        y_ptr: Bx::Value,
        tmp_ty: Bx::Type,
        access_align: Align,
    ) {
        // FIXME: Remove pointercast when stop support of LLVM 14.
        let tmp_ptr_ty = bx.type_ptr_to(tmp_ty);
        let x_ptr = bx.pointercast(x_ptr, tmp_ptr_ty);
        let y_ptr = bx.pointercast(y_ptr, tmp_ptr_ty);

        let tmp_x = bx.load(tmp_ty, x_ptr, access_align);
        let tmp_y = bx.load(tmp_ty, y_ptr, access_align);
        bx.store(tmp_y, x_ptr, access_align);
        bx.store(tmp_x, y_ptr, access_align);
    }

    fn swap_using_single_temp<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        x_ptr: Bx::Value,
        y_ptr: Bx::Value,
        tmp_ty: Bx::Type,
        tmp_size: Size,
        access_align: Align,
    ) {
        // FIXME: Remove pointercast when stop support of LLVM 14.
        let tmp_ptr_ty = bx.type_ptr_to(tmp_ty);
        let x_ptr = bx.pointercast(x_ptr, tmp_ptr_ty);
        let y_ptr = bx.pointercast(y_ptr, tmp_ptr_ty);

        let num_bytes = bx.const_usize(tmp_size.bytes());
        let tmp_x = bx.load(tmp_ty, x_ptr, access_align);
        // FIXME(AngelicosPhosphoros): Use memcpy.inline here.
        bx.memcpy(x_ptr, access_align, y_ptr, access_align, num_bytes, MemFlags::empty());
        bx.store(tmp_x, y_ptr, access_align);
    }
}

fn memset_intrinsic<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    volatile: bool,
    ty: Ty<'tcx>,
    dst: Bx::Value,
    val: Bx::Value,
    count: Bx::Value,
) {
    let layout = bx.layout_of(ty);
    let size = layout.size;
    let align = layout.align.abi;
    let size = bx.mul(bx.const_usize(size.bytes()), count);
    let flags = if volatile { MemFlags::VOLATILE } else { MemFlags::empty() };
    bx.memset(dst, val, size, align, flags);
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_intrinsic_call(
        bx: &mut Bx,
        instance: ty::Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Bx::Value>],
        llresult: Bx::Value,
        span: Span,
    ) {
        let callee_ty = instance.ty(bx.tcx(), ty::ParamEnv::reveal_all());

        let ty::FnDef(def_id, substs) = *callee_ty.kind() else {
            bug!("expected fn item type, found {}", callee_ty);
        };

        let sig = callee_ty.fn_sig(bx.tcx());
        let sig = bx.tcx().normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = bx.tcx().item_name(def_id);
        let name_str = name.as_str();

        let llret_ty = bx.backend_type(bx.layout_of(ret_ty));
        let result = PlaceRef::new_sized(llresult, fn_abi.ret.layout);

        let llval = match name {
            sym::abort => {
                bx.abort();
                return;
            }

            sym::va_start => bx.va_start(args[0].immediate()),
            sym::va_end => bx.va_end(args[0].immediate()),
            sym::size_of_val => {
                let tp_ty = substs.type_at(0);
                if let OperandValue::Pair(_, meta) = args[0].val {
                    let (llsize, _) = glue::size_and_align_of_dst(bx, tp_ty, Some(meta));
                    llsize
                } else {
                    bx.const_usize(bx.layout_of(tp_ty).size.bytes())
                }
            }
            sym::min_align_of_val => {
                let tp_ty = substs.type_at(0);
                if let OperandValue::Pair(_, meta) = args[0].val {
                    let (_, llalign) = glue::size_and_align_of_dst(bx, tp_ty, Some(meta));
                    llalign
                } else {
                    bx.const_usize(bx.layout_of(tp_ty).align.abi.bytes())
                }
            }
            sym::vtable_size | sym::vtable_align => {
                let vtable = args[0].immediate();
                let idx = match name {
                    sym::vtable_size => ty::COMMON_VTABLE_ENTRIES_SIZE,
                    sym::vtable_align => ty::COMMON_VTABLE_ENTRIES_ALIGN,
                    _ => bug!(),
                };
                let value = meth::VirtualIndex::from_index(idx).get_usize(bx, vtable);
                match name {
                    // Size is always <= isize::MAX.
                    sym::vtable_size => {
                        let size_bound = bx.data_layout().ptr_sized_integer().signed_max() as u128;
                        bx.range_metadata(value, WrappingRange { start: 0, end: size_bound });
                    },
                    // Alignment is always nonzero.
                    sym::vtable_align => bx.range_metadata(value, WrappingRange { start: 1, end: !0 }),
                    _ => {}
                }
                value
            }
            sym::pref_align_of
            | sym::needs_drop
            | sym::type_id
            | sym::type_name
            | sym::variant_count => {
                let value = bx
                    .tcx()
                    .const_eval_instance(ty::ParamEnv::reveal_all(), instance, None)
                    .unwrap();
                OperandRef::from_const(bx, value, ret_ty).immediate_or_packed_pair(bx)
            }
            sym::arith_offset => {
                let ty = substs.type_at(0);
                let layout = bx.layout_of(ty);
                let ptr = args[0].immediate();
                let offset = args[1].immediate();
                bx.gep(bx.backend_type(layout), ptr, &[offset])
            }
            sym::copy => {
                copy_intrinsic(
                    bx,
                    true,
                    false,
                    substs.type_at(0),
                    args[1].immediate(),
                    args[0].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::swap_nonoverlapping_single => {
                swap_intrinsic::single(
                    bx,
                    args[0].immediate(),
                    args[1].immediate(),
                    substs.type_at(0),
                    span,
                );
                return;
            }
            sym::swap_nonoverlapping_many => {
                swap_intrinsic::many(
                    bx,
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                    substs.type_at(0),
                    span,
                );
                return;
            }
            sym::write_bytes => {
                memset_intrinsic(
                    bx,
                    false,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }

            sym::volatile_copy_nonoverlapping_memory => {
                copy_intrinsic(
                    bx,
                    false,
                    true,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::volatile_copy_memory => {
                copy_intrinsic(
                    bx,
                    true,
                    true,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::volatile_set_memory => {
                memset_intrinsic(
                    bx,
                    true,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::volatile_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.volatile_store(bx, dst);
                return;
            }
            sym::unaligned_volatile_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.unaligned_volatile_store(bx, dst);
                return;
            }
            | sym::unchecked_shl
            | sym::unchecked_shr
            | sym::unchecked_add
            | sym::unchecked_sub
            | sym::unchecked_mul
            | sym::exact_div => {
                let ty = arg_tys[0];
                match int_type_width_signed(ty, bx.tcx()) {
                    Some((_width, signed)) => match name {
                        sym::exact_div => {
                            if signed {
                                bx.exactsdiv(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.exactudiv(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_shl => bx.shl(args[0].immediate(), args[1].immediate()),
                        sym::unchecked_shr => {
                            if signed {
                                bx.ashr(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.lshr(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_add => {
                            if signed {
                                bx.unchecked_sadd(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.unchecked_uadd(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_sub => {
                            if signed {
                                bx.unchecked_ssub(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.unchecked_usub(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_mul => {
                            if signed {
                                bx.unchecked_smul(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.unchecked_umul(args[0].immediate(), args[1].immediate())
                            }
                        }
                        _ => bug!(),
                    },
                    None => {
                        bx.tcx().sess.emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
                        return;
                    }
                }
            }
            sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
                match float_type_width(arg_tys[0]) {
                    Some(_width) => match name {
                        sym::fadd_fast => bx.fadd_fast(args[0].immediate(), args[1].immediate()),
                        sym::fsub_fast => bx.fsub_fast(args[0].immediate(), args[1].immediate()),
                        sym::fmul_fast => bx.fmul_fast(args[0].immediate(), args[1].immediate()),
                        sym::fdiv_fast => bx.fdiv_fast(args[0].immediate(), args[1].immediate()),
                        sym::frem_fast => bx.frem_fast(args[0].immediate(), args[1].immediate()),
                        _ => bug!(),
                    },
                    None => {
                        bx.tcx().sess.emit_err(InvalidMonomorphization::BasicFloatType { span, name, ty: arg_tys[0] });
                        return;
                    }
                }
            }

            sym::float_to_int_unchecked => {
                if float_type_width(arg_tys[0]).is_none() {
                    bx.tcx().sess.emit_err(InvalidMonomorphization::FloatToIntUnchecked { span, ty: arg_tys[0] });
                    return;
                }
                let Some((_width, signed)) = int_type_width_signed(ret_ty, bx.tcx()) else {
                    bx.tcx().sess.emit_err(InvalidMonomorphization::FloatToIntUnchecked { span, ty: ret_ty });
                    return;
                };
                if signed {
                    bx.fptosi(args[0].immediate(), llret_ty)
                } else {
                    bx.fptoui(args[0].immediate(), llret_ty)
                }
            }

            sym::discriminant_value => {
                if ret_ty.is_integral() {
                    args[0].deref(bx.cx()).codegen_get_discr(bx, ret_ty)
                } else {
                    span_bug!(span, "Invalid discriminant type for `{:?}`", arg_tys[0])
                }
            }

            sym::const_allocate => {
                // returns a null pointer at runtime.
                bx.const_null(bx.type_i8p())
            }

            sym::const_deallocate => {
                // nop at runtime.
                return;
            }

            // This requires that atomic intrinsics follow a specific naming pattern:
            // "atomic_<operation>[_<ordering>]"
            name if let Some(atomic) = name_str.strip_prefix("atomic_") => {
                use crate::common::AtomicOrdering::*;
                use crate::common::{AtomicRmwBinOp, SynchronizationScope};

                let Some((instruction, ordering)) = atomic.split_once('_') else {
                    bx.sess().emit_fatal(errors::MissingMemoryOrdering);
                };

                let parse_ordering = |bx: &Bx, s| match s {
                    "unordered" => Unordered,
                    "relaxed" => Relaxed,
                    "acquire" => Acquire,
                    "release" => Release,
                    "acqrel" => AcquireRelease,
                    "seqcst" => SequentiallyConsistent,
                    _ => bx.sess().emit_fatal(errors::UnknownAtomicOrdering),
                };

                let invalid_monomorphization = |ty| {
                    bx.tcx().sess.emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
                };

                match instruction {
                    "cxchg" | "cxchgweak" => {
                        let Some((success, failure)) = ordering.split_once('_') else {
                            bx.sess().emit_fatal(errors::AtomicCompareExchange);
                        };
                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let weak = instruction == "cxchgweak";
                            let mut dst = args[0].immediate();
                            let mut cmp = args[1].immediate();
                            let mut src = args[2].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first.
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                dst = bx.pointercast(dst, ptr_llty);
                                cmp = bx.ptrtoint(cmp, bx.type_isize());
                                src = bx.ptrtoint(src, bx.type_isize());
                            }
                            let pair = bx.atomic_cmpxchg(dst, cmp, src, parse_ordering(bx, success), parse_ordering(bx, failure), weak);
                            let val = bx.extract_value(pair, 0);
                            let success = bx.extract_value(pair, 1);
                            let val = bx.from_immediate(val);
                            let success = bx.from_immediate(success);

                            let dest = result.project_field(bx, 0);
                            bx.store(val, dest.llval, dest.align);
                            let dest = result.project_field(bx, 1);
                            bx.store(success, dest.llval, dest.align);
                            return;
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "load" => {
                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let layout = bx.layout_of(ty);
                            let size = layout.size;
                            let mut source = args[0].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first...
                                let llty = bx.type_isize();
                                let ptr_llty = bx.type_ptr_to(llty);
                                source = bx.pointercast(source, ptr_llty);
                                let result = bx.atomic_load(llty, source, parse_ordering(bx, ordering), size);
                                // ... and then cast the result back to a pointer
                                bx.inttoptr(result, bx.backend_type(layout))
                            } else {
                                bx.atomic_load(bx.backend_type(layout), source, parse_ordering(bx, ordering), size)
                            }
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "store" => {
                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let size = bx.layout_of(ty).size;
                            let mut val = args[1].immediate();
                            let mut ptr = args[0].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first.
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                ptr = bx.pointercast(ptr, ptr_llty);
                                val = bx.ptrtoint(val, bx.type_isize());
                            }
                            bx.atomic_store(val, ptr, parse_ordering(bx, ordering), size);
                            return;
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "fence" => {
                        bx.atomic_fence(parse_ordering(bx, ordering), SynchronizationScope::CrossThread);
                        return;
                    }

                    "singlethreadfence" => {
                        bx.atomic_fence(parse_ordering(bx, ordering), SynchronizationScope::SingleThread);
                        return;
                    }

                    // These are all AtomicRMW ops
                    op => {
                        let atom_op = match op {
                            "xchg" => AtomicRmwBinOp::AtomicXchg,
                            "xadd" => AtomicRmwBinOp::AtomicAdd,
                            "xsub" => AtomicRmwBinOp::AtomicSub,
                            "and" => AtomicRmwBinOp::AtomicAnd,
                            "nand" => AtomicRmwBinOp::AtomicNand,
                            "or" => AtomicRmwBinOp::AtomicOr,
                            "xor" => AtomicRmwBinOp::AtomicXor,
                            "max" => AtomicRmwBinOp::AtomicMax,
                            "min" => AtomicRmwBinOp::AtomicMin,
                            "umax" => AtomicRmwBinOp::AtomicUMax,
                            "umin" => AtomicRmwBinOp::AtomicUMin,
                            _ => bx.sess().emit_fatal(errors::UnknownAtomicOperation),
                        };

                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let mut ptr = args[0].immediate();
                            let mut val = args[1].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first.
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                ptr = bx.pointercast(ptr, ptr_llty);
                                val = bx.ptrtoint(val, bx.type_isize());
                            }
                            bx.atomic_rmw(atom_op, ptr, val, parse_ordering(bx, ordering))
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }
                }
            }

            sym::nontemporal_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.nontemporal_store(bx, dst);
                return;
            }

            sym::ptr_guaranteed_cmp => {
                let a = args[0].immediate();
                let b = args[1].immediate();
                bx.icmp(IntPredicate::IntEQ, a, b)
            }

            sym::ptr_offset_from | sym::ptr_offset_from_unsigned => {
                let ty = substs.type_at(0);
                let pointee_size = bx.layout_of(ty).size;

                let a = args[0].immediate();
                let b = args[1].immediate();
                let a = bx.ptrtoint(a, bx.type_isize());
                let b = bx.ptrtoint(b, bx.type_isize());
                let pointee_size = bx.const_usize(pointee_size.bytes());
                if name == sym::ptr_offset_from {
                    // This is the same sequence that Clang emits for pointer subtraction.
                    // It can be neither `nsw` nor `nuw` because the input is treated as
                    // unsigned but then the output is treated as signed, so neither works.
                    let d = bx.sub(a, b);
                    // this is where the signed magic happens (notice the `s` in `exactsdiv`)
                    bx.exactsdiv(d, pointee_size)
                } else {
                    // The `_unsigned` version knows the relative ordering of the pointers,
                    // so can use `sub nuw` and `udiv exact` instead of dealing in signed.
                    let d = bx.unchecked_usub(a, b);
                    bx.exactudiv(d, pointee_size)
                }
            }

            _ => {
                // Need to use backend-specific things in the implementation.
                bx.codegen_intrinsic_call(instance, fn_abi, args, llresult, span);
                return;
            }
        };

        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast(ty, _) = &fn_abi.ret.mode {
                let ptr_llty = bx.type_ptr_to(bx.cast_backend_type(ty));
                let ptr = bx.pointercast(result.llval, ptr_llty);
                bx.store(llval, ptr, result.align);
            } else {
                OperandRef::from_immediate_or_packed_pair(bx, llval, result.layout)
                    .val
                    .store(bx, result);
            }
        }
    }
}

// Returns the width of an int Ty, and if it's signed or not
// Returns None if the type is not an integer
// FIXME: there’s multiple of this functions, investigate using some of the already existing
// stuffs.
fn int_type_width_signed(ty: Ty<'_>, tcx: TyCtxt<'_>) -> Option<(u64, bool)> {
    match ty.kind() {
        ty::Int(t) => {
            Some((t.bit_width().unwrap_or(u64::from(tcx.sess.target.pointer_width)), true))
        }
        ty::Uint(t) => {
            Some((t.bit_width().unwrap_or(u64::from(tcx.sess.target.pointer_width)), false))
        }
        _ => None,
    }
}

// Returns the width of a float Ty
// Returns None if the type is not a float
fn float_type_width(ty: Ty<'_>) -> Option<u64> {
    match ty.kind() {
        ty::Float(t) => Some(t.bit_width()),
        _ => None,
    }
}
