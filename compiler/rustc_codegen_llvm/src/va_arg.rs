use rustc_abi::{Align, Endian, HasDataLayout, Size};
use rustc_codegen_ssa::common::IntPredicate;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods};
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf};

use crate::builder::Builder;
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;

fn round_up_to_alignment<'ll>(
    bx: &mut Builder<'_, 'll, '_>,
    mut value: &'ll Value,
    align: Align,
) -> &'ll Value {
    value = bx.add(value, bx.cx().const_i32(align.bytes() as i32 - 1));
    return bx.and(value, bx.cx().const_i32(-(align.bytes() as i32)));
}

fn round_pointer_up_to_alignment<'ll>(
    bx: &mut Builder<'_, 'll, '_>,
    addr: &'ll Value,
    align: Align,
    ptr_ty: &'ll Type,
) -> &'ll Value {
    let mut ptr_as_int = bx.ptrtoint(addr, bx.cx().type_isize());
    ptr_as_int = round_up_to_alignment(bx, ptr_as_int, align);
    bx.inttoptr(ptr_as_int, ptr_ty)
}

fn emit_direct_ptr_va_arg<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    size: Size,
    align: Align,
    slot_size: Align,
    allow_higher_align: bool,
) -> (&'ll Value, Align) {
    let va_list_ty = bx.type_ptr();
    let va_list_addr = list.immediate();

    let ptr = bx.load(va_list_ty, va_list_addr, bx.tcx().data_layout.pointer_align.abi);

    let (addr, addr_align) = if allow_higher_align && align > slot_size {
        (round_pointer_up_to_alignment(bx, ptr, align, bx.type_ptr()), align)
    } else {
        (ptr, slot_size)
    };

    let aligned_size = size.align_to(slot_size).bytes() as i32;
    let full_direct_size = bx.cx().const_i32(aligned_size);
    let next = bx.inbounds_ptradd(addr, full_direct_size);
    bx.store(next, va_list_addr, bx.tcx().data_layout.pointer_align.abi);

    if size.bytes() < slot_size.bytes() && bx.tcx().sess.target.endian == Endian::Big {
        let adjusted_size = bx.cx().const_i32((slot_size.bytes() - size.bytes()) as i32);
        let adjusted = bx.inbounds_ptradd(addr, adjusted_size);
        (adjusted, addr_align)
    } else {
        (addr, addr_align)
    }
}

enum PassMode {
    Direct,
    Indirect,
}

enum SlotSize {
    Bytes8 = 8,
    Bytes4 = 4,
}

enum AllowHigherAlign {
    No,
    Yes,
}

fn emit_ptr_va_arg<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
    pass_mode: PassMode,
    slot_size: SlotSize,
    allow_higher_align: AllowHigherAlign,
) -> &'ll Value {
    let indirect = matches!(pass_mode, PassMode::Indirect);
    let allow_higher_align = matches!(allow_higher_align, AllowHigherAlign::Yes);
    let slot_size = Align::from_bytes(slot_size as u64).unwrap();

    let layout = bx.cx.layout_of(target_ty);
    let (llty, size, align) = if indirect {
        (
            bx.cx.layout_of(Ty::new_imm_ptr(bx.cx.tcx, target_ty)).llvm_type(bx.cx),
            bx.cx.data_layout().pointer_size,
            bx.cx.data_layout().pointer_align,
        )
    } else {
        (layout.llvm_type(bx.cx), layout.size, layout.align)
    };
    let (addr, addr_align) =
        emit_direct_ptr_va_arg(bx, list, size, align.abi, slot_size, allow_higher_align);
    if indirect {
        let tmp_ret = bx.load(llty, addr, addr_align);
        bx.load(bx.cx.layout_of(target_ty).llvm_type(bx.cx), tmp_ret, align.abi)
    } else {
        bx.load(llty, addr, addr_align)
    }
}

fn emit_aapcs_va_arg<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    let dl = bx.cx.data_layout();

    // Implementation of the AAPCS64 calling convention for va_args see
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst
    //
    // typedef struct  va_list {
    //     void * stack; // next stack param
    //     void * gr_top; // end of GP arg reg save area
    //     void * vr_top; // end of FP/SIMD arg reg save area
    //     int gr_offs; // offset from  gr_top to next GP register arg
    //     int vr_offs; // offset from  vr_top to next FP/SIMD register arg
    // } va_list;
    let va_list_addr = list.immediate();

    // There is no padding between fields since `void*` is size=8 align=8, `int` is size=4 align=4.
    // See https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst
    // Table 1, Byte size and byte alignment of fundamental data types
    // Table 3, Mapping of C & C++ built-in data types
    let ptr_offset = 8;
    let i32_offset = 4;
    let gr_top = bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(ptr_offset));
    let vr_top = bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(2 * ptr_offset));
    let gr_offs = bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(3 * ptr_offset));
    let vr_offs = bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(3 * ptr_offset + i32_offset));

    let layout = bx.cx.layout_of(target_ty);

    let maybe_reg = bx.append_sibling_block("va_arg.maybe_reg");
    let in_reg = bx.append_sibling_block("va_arg.in_reg");
    let on_stack = bx.append_sibling_block("va_arg.on_stack");
    let end = bx.append_sibling_block("va_arg.end");
    let zero = bx.const_i32(0);
    let offset_align = Align::from_bytes(4).unwrap();

    let gr_type = target_ty.is_any_ptr() || target_ty.is_integral();
    let (reg_off, reg_top, slot_size) = if gr_type {
        let nreg = (layout.size.bytes() + 7) / 8;
        (gr_offs, gr_top, nreg * 8)
    } else {
        let nreg = (layout.size.bytes() + 15) / 16;
        (vr_offs, vr_top, nreg * 16)
    };

    // if the offset >= 0 then the value will be on the stack
    let mut reg_off_v = bx.load(bx.type_i32(), reg_off, offset_align);
    let use_stack = bx.icmp(IntPredicate::IntSGE, reg_off_v, zero);
    bx.cond_br(use_stack, on_stack, maybe_reg);

    // The value at this point might be in a register, but there is a chance that
    // it could be on the stack so we have to update the offset and then check
    // the offset again.

    bx.switch_to_block(maybe_reg);
    if gr_type && layout.align.abi.bytes() > 8 {
        reg_off_v = bx.add(reg_off_v, bx.const_i32(15));
        reg_off_v = bx.and(reg_off_v, bx.const_i32(-16));
    }
    let new_reg_off_v = bx.add(reg_off_v, bx.const_i32(slot_size as i32));

    bx.store(new_reg_off_v, reg_off, offset_align);

    // Check to see if we have overflowed the registers as a result of this.
    // If we have then we need to use the stack for this value
    let use_stack = bx.icmp(IntPredicate::IntSGT, new_reg_off_v, zero);
    bx.cond_br(use_stack, on_stack, in_reg);

    bx.switch_to_block(in_reg);
    let top_type = bx.type_ptr();
    let top = bx.load(top_type, reg_top, dl.pointer_align.abi);

    // reg_value = *(@top + reg_off_v);
    let mut reg_addr = bx.ptradd(top, reg_off_v);
    if bx.tcx().sess.target.endian == Endian::Big && layout.size.bytes() != slot_size {
        // On big-endian systems the value is right-aligned in its slot.
        let offset = bx.const_i32((slot_size - layout.size.bytes()) as i32);
        reg_addr = bx.ptradd(reg_addr, offset);
    }
    let reg_type = layout.llvm_type(bx);
    let reg_value = bx.load(reg_type, reg_addr, layout.align.abi);
    bx.br(end);

    // On Stack block
    bx.switch_to_block(on_stack);
    let stack_value = emit_ptr_va_arg(
        bx,
        list,
        target_ty,
        PassMode::Direct,
        SlotSize::Bytes8,
        AllowHigherAlign::Yes,
    );
    bx.br(end);

    bx.switch_to_block(end);
    let val =
        bx.phi(layout.immediate_llvm_type(bx), &[reg_value, stack_value], &[in_reg, on_stack]);

    val
}

fn emit_s390x_va_arg<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    let dl = bx.cx.data_layout();

    // Implementation of the s390x ELF ABI calling convention for va_args see
    // https://github.com/IBM/s390x-abi (chapter 1.2.4)
    //
    // typedef struct __va_list_tag {
    //     long __gpr;
    //     long __fpr;
    //     void *__overflow_arg_area;
    //     void *__reg_save_area;
    // } va_list[1];
    let va_list_addr = list.immediate();

    // There is no padding between fields since `long` and `void*` both have size=8 align=8.
    // https://github.com/IBM/s390x-abi (Table 1.1.: Scalar types)
    let i64_offset = 8;
    let ptr_offset = 8;
    let gpr = va_list_addr;
    let fpr = bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(i64_offset));
    let overflow_arg_area = bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(2 * i64_offset));
    let reg_save_area =
        bx.inbounds_ptradd(va_list_addr, bx.cx.const_usize(2 * i64_offset + ptr_offset));

    let layout = bx.cx.layout_of(target_ty);

    let in_reg = bx.append_sibling_block("va_arg.in_reg");
    let in_mem = bx.append_sibling_block("va_arg.in_mem");
    let end = bx.append_sibling_block("va_arg.end");

    // FIXME: vector ABI not yet supported.
    let target_ty_size = bx.cx.size_of(target_ty).bytes();
    let indirect: bool = target_ty_size > 8 || !target_ty_size.is_power_of_two();
    let unpadded_size = if indirect { 8 } else { target_ty_size };
    let padded_size = 8;
    let padding = padded_size - unpadded_size;

    let gpr_type = indirect || !layout.is_single_fp_element(bx.cx);
    let (max_regs, reg_count, reg_save_index, reg_padding) =
        if gpr_type { (5, gpr, 2, padding) } else { (4, fpr, 16, 0) };

    // Check whether the value was passed in a register or in memory.
    let reg_count_v = bx.load(bx.type_i64(), reg_count, Align::from_bytes(8).unwrap());
    let use_regs = bx.icmp(IntPredicate::IntULT, reg_count_v, bx.const_u64(max_regs));
    bx.cond_br(use_regs, in_reg, in_mem);

    // Emit code to load the value if it was passed in a register.
    bx.switch_to_block(in_reg);

    // Work out the address of the value in the register save area.
    let reg_ptr_v = bx.load(bx.type_ptr(), reg_save_area, dl.pointer_align.abi);
    let scaled_reg_count = bx.mul(reg_count_v, bx.const_u64(8));
    let reg_off = bx.add(scaled_reg_count, bx.const_u64(reg_save_index * 8 + reg_padding));
    let reg_addr = bx.ptradd(reg_ptr_v, reg_off);

    // Update the register count.
    let new_reg_count_v = bx.add(reg_count_v, bx.const_u64(1));
    bx.store(new_reg_count_v, reg_count, Align::from_bytes(8).unwrap());
    bx.br(end);

    // Emit code to load the value if it was passed in memory.
    bx.switch_to_block(in_mem);

    // Work out the address of the value in the argument overflow area.
    let arg_ptr_v =
        bx.load(bx.type_ptr(), overflow_arg_area, bx.tcx().data_layout.pointer_align.abi);
    let arg_off = bx.const_u64(padding);
    let mem_addr = bx.ptradd(arg_ptr_v, arg_off);

    // Update the argument overflow area pointer.
    let arg_size = bx.cx().const_u64(padded_size);
    let new_arg_ptr_v = bx.inbounds_ptradd(arg_ptr_v, arg_size);
    bx.store(new_arg_ptr_v, overflow_arg_area, dl.pointer_align.abi);
    bx.br(end);

    // Return the appropriate result.
    bx.switch_to_block(end);
    let val_addr = bx.phi(bx.type_ptr(), &[reg_addr, mem_addr], &[in_reg, in_mem]);
    let val_type = layout.llvm_type(bx);
    let val_addr =
        if indirect { bx.load(bx.cx.type_ptr(), val_addr, dl.pointer_align.abi) } else { val_addr };
    bx.load(val_type, val_addr, layout.align.abi)
}

fn emit_xtensa_va_arg<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    // Implementation of va_arg for Xtensa. There doesn't seem to be an authoritative source for
    // this, other than "what GCC does".
    //
    // The va_list type has three fields:
    // struct __va_list_tag {
    //   int32_t *va_stk; // Arguments passed on the stack
    //   int32_t *va_reg; // Arguments passed in registers, saved to memory by the prologue.
    //   int32_t va_ndx; // Offset into the arguments, in bytes
    // };
    //
    // The first 24 bytes (equivalent to 6 registers) come from va_reg, the rest from va_stk.
    // Thus if va_ndx is less than 24, the next va_arg *may* read from va_reg,
    // otherwise it must come from va_stk.
    //
    // Primitive arguments are never split between registers and the stack. For example, if loading an 8 byte
    // primitive value and va_ndx = 20, we instead bump the offset and read everything from va_stk.
    let va_list_addr = list.immediate();
    // FIXME: handle multi-field structs that split across regsave/stack?
    let layout = bx.cx.layout_of(target_ty);
    let from_stack = bx.append_sibling_block("va_arg.from_stack");
    let from_regsave = bx.append_sibling_block("va_arg.from_regsave");
    let end = bx.append_sibling_block("va_arg.end");

    // (*va).va_ndx
    let va_reg_offset = 4;
    let va_ndx_offset = va_reg_offset + 4;
    let offset_ptr =
        bx.inbounds_gep(bx.type_i8(), va_list_addr, &[bx.cx.const_usize(va_ndx_offset)]);

    let offset = bx.load(bx.type_i32(), offset_ptr, bx.tcx().data_layout.i32_align.abi);
    let offset = round_up_to_alignment(bx, offset, layout.align.abi);

    let slot_size = layout.size.align_to(Align::from_bytes(4).unwrap()).bytes() as i32;

    // Update the offset in va_list, by adding the slot's size.
    let offset_next = bx.add(offset, bx.const_i32(slot_size));

    // Figure out where to look for our value. We do that by checking the end of our slot (offset_next).
    // If that is within the regsave area, then load from there. Otherwise load from the stack area.
    let regsave_size = bx.const_i32(24);
    let use_regsave = bx.icmp(IntPredicate::IntULE, offset_next, regsave_size);
    bx.cond_br(use_regsave, from_regsave, from_stack);

    bx.switch_to_block(from_regsave);
    // update va_ndx
    bx.store(offset_next, offset_ptr, bx.tcx().data_layout.pointer_align.abi);

    // (*va).va_reg
    let regsave_area_ptr =
        bx.inbounds_gep(bx.type_i8(), va_list_addr, &[bx.cx.const_usize(va_reg_offset)]);
    let regsave_area =
        bx.load(bx.type_ptr(), regsave_area_ptr, bx.tcx().data_layout.pointer_align.abi);
    let regsave_value_ptr = bx.inbounds_gep(bx.type_i8(), regsave_area, &[offset]);
    bx.br(end);

    bx.switch_to_block(from_stack);

    // The first time we switch from regsave to stack we needs to adjust our offsets a bit.
    // va_stk is set up such that the first stack argument is always at va_stk + 32.
    // The corrected offset is written back into the va_list struct.

    // let offset_corrected = cmp::max(offset, 32);
    let stack_offset_start = bx.const_i32(32);
    let needs_correction = bx.icmp(IntPredicate::IntULE, offset, stack_offset_start);
    let offset_corrected = bx.select(needs_correction, stack_offset_start, offset);

    // let offset_next_corrected = offset_corrected + slot_size;
    // va_ndx = offset_next_corrected;
    let offset_next_corrected = bx.add(offset_next, bx.const_i32(slot_size));
    // update va_ndx
    bx.store(offset_next_corrected, offset_ptr, bx.tcx().data_layout.pointer_align.abi);

    // let stack_value_ptr = unsafe { (*va).va_stk.byte_add(offset_corrected) };
    let stack_area_ptr = bx.inbounds_gep(bx.type_i8(), va_list_addr, &[bx.cx.const_usize(0)]);
    let stack_area = bx.load(bx.type_ptr(), stack_area_ptr, bx.tcx().data_layout.pointer_align.abi);
    let stack_value_ptr = bx.inbounds_gep(bx.type_i8(), stack_area, &[offset_corrected]);
    bx.br(end);

    bx.switch_to_block(end);

    // On big-endian, for values smaller than the slot size we'd have to align the read to the end
    // of the slot rather than the start. While the ISA and GCC support big-endian, all the Xtensa
    // targets supported by rustc are litte-endian so don't worry about it.

    // if from_regsave {
    //     unsafe { *regsave_value_ptr }
    // } else {
    //     unsafe { *stack_value_ptr }
    // }
    assert!(bx.tcx().sess.target.endian == Endian::Little);
    let value_ptr =
        bx.phi(bx.type_ptr(), &[regsave_value_ptr, stack_value_ptr], &[from_regsave, from_stack]);
    return bx.load(layout.llvm_type(bx), value_ptr, layout.align.abi);
}

pub(super) fn emit_va_arg<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    addr: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    // Determine the va_arg implementation to use. The LLVM va_arg instruction
    // is lacking in some instances, so we should only use it as a fallback.
    let target = &bx.cx.tcx.sess.target;

    match &*target.arch {
        "x86" => emit_ptr_va_arg(
            bx,
            addr,
            target_ty,
            PassMode::Direct,
            SlotSize::Bytes4,
            if target.is_like_windows { AllowHigherAlign::No } else { AllowHigherAlign::Yes },
        ),
        "aarch64" | "arm64ec" if target.is_like_windows || target.is_like_darwin => {
            emit_ptr_va_arg(
                bx,
                addr,
                target_ty,
                PassMode::Direct,
                SlotSize::Bytes8,
                if target.is_like_windows { AllowHigherAlign::No } else { AllowHigherAlign::Yes },
            )
        }
        "aarch64" => emit_aapcs_va_arg(bx, addr, target_ty),
        "s390x" => emit_s390x_va_arg(bx, addr, target_ty),
        // Windows x86_64
        "x86_64" if target.is_like_windows => {
            let target_ty_size = bx.cx.size_of(target_ty).bytes();
            emit_ptr_va_arg(
                bx,
                addr,
                target_ty,
                if target_ty_size > 8 || !target_ty_size.is_power_of_two() {
                    PassMode::Indirect
                } else {
                    PassMode::Direct
                },
                SlotSize::Bytes8,
                AllowHigherAlign::No,
            )
        }
        "xtensa" => emit_xtensa_va_arg(bx, addr, target_ty),
        // For all other architecture/OS combinations fall back to using
        // the LLVM va_arg instruction.
        // https://llvm.org/docs/LangRef.html#va-arg-instruction
        _ => bx.va_arg(addr.immediate(), bx.cx.layout_of(target_ty).llvm_type(bx.cx)),
    }
}
