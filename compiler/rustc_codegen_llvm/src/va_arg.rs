use crate::builder::Builder;
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::{
    common::IntPredicate,
    traits::{BaseTypeMethods, BuilderMethods, ConstMethods, DerivedTypeMethods},
};
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::Ty;
use rustc_target::abi::{Align, HasDataLayout, LayoutOf, Size};

fn round_pointer_up_to_alignment(
    bx: &mut Builder<'a, 'll, 'tcx>,
    addr: &'ll Value,
    align: Align,
    ptr_ty: &'ll Type,
) -> &'ll Value {
    let mut ptr_as_int = bx.ptrtoint(addr, bx.cx().type_isize());
    ptr_as_int = bx.add(ptr_as_int, bx.cx().const_i32(align.bytes() as i32 - 1));
    ptr_as_int = bx.and(ptr_as_int, bx.cx().const_i32(-(align.bytes() as i32)));
    bx.inttoptr(ptr_as_int, ptr_ty)
}

fn emit_direct_ptr_va_arg(
    bx: &mut Builder<'a, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    llty: &'ll Type,
    size: Size,
    align: Align,
    slot_size: Align,
    allow_higher_align: bool,
) -> (&'ll Value, Align) {
    let va_list_ptr_ty = bx.cx().type_ptr_to(bx.cx.type_i8p());
    let va_list_addr = if list.layout.llvm_type(bx.cx) != va_list_ptr_ty {
        bx.bitcast(list.immediate(), va_list_ptr_ty)
    } else {
        list.immediate()
    };

    let ptr = bx.load(va_list_addr, bx.tcx().data_layout.pointer_align.abi);

    let (addr, addr_align) = if allow_higher_align && align > slot_size {
        (round_pointer_up_to_alignment(bx, ptr, align, bx.cx().type_i8p()), align)
    } else {
        (ptr, slot_size)
    };

    let aligned_size = size.align_to(slot_size).bytes() as i32;
    let full_direct_size = bx.cx().const_i32(aligned_size);
    let next = bx.inbounds_gep(addr, &[full_direct_size]);
    bx.store(next, va_list_addr, bx.tcx().data_layout.pointer_align.abi);

    if size.bytes() < slot_size.bytes() && &*bx.tcx().sess.target.target_endian == "big" {
        let adjusted_size = bx.cx().const_i32((slot_size.bytes() - size.bytes()) as i32);
        let adjusted = bx.inbounds_gep(addr, &[adjusted_size]);
        (bx.bitcast(adjusted, bx.cx().type_ptr_to(llty)), addr_align)
    } else {
        (bx.bitcast(addr, bx.cx().type_ptr_to(llty)), addr_align)
    }
}

fn emit_ptr_va_arg(
    bx: &mut Builder<'a, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
    indirect: bool,
    slot_size: Align,
    allow_higher_align: bool,
) -> &'ll Value {
    let layout = bx.cx.layout_of(target_ty);
    let (llty, size, align) = if indirect {
        (
            bx.cx.layout_of(bx.cx.tcx.mk_imm_ptr(target_ty)).llvm_type(bx.cx),
            bx.cx.data_layout().pointer_size,
            bx.cx.data_layout().pointer_align,
        )
    } else {
        (layout.llvm_type(bx.cx), layout.size, layout.align)
    };
    let (addr, addr_align) =
        emit_direct_ptr_va_arg(bx, list, llty, size, align.abi, slot_size, allow_higher_align);
    if indirect {
        let tmp_ret = bx.load(addr, addr_align);
        bx.load(tmp_ret, align.abi)
    } else {
        bx.load(addr, addr_align)
    }
}

fn emit_aapcs_va_arg(
    bx: &mut Builder<'a, 'll, 'tcx>,
    list: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    // Implementation of the AAPCS64 calling convention for va_args see
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst
    let va_list_addr = list.immediate();
    let layout = bx.cx.layout_of(target_ty);

    let mut maybe_reg = bx.build_sibling_block("va_arg.maybe_reg");
    let mut in_reg = bx.build_sibling_block("va_arg.in_reg");
    let mut on_stack = bx.build_sibling_block("va_arg.on_stack");
    let mut end = bx.build_sibling_block("va_arg.end");
    let zero = bx.const_i32(0);
    let offset_align = Align::from_bytes(4).unwrap();
    assert!(&*bx.tcx().sess.target.target_endian == "little");

    let gr_type = target_ty.is_any_ptr() || target_ty.is_integral();
    let (reg_off, reg_top_index, slot_size) = if gr_type {
        let gr_offs = bx.struct_gep(va_list_addr, 7);
        let nreg = (layout.size.bytes() + 7) / 8;
        (gr_offs, 3, nreg * 8)
    } else {
        let vr_off = bx.struct_gep(va_list_addr, 9);
        let nreg = (layout.size.bytes() + 15) / 16;
        (vr_off, 5, nreg * 16)
    };

    // if the offset >= 0 then the value will be on the stack
    let mut reg_off_v = bx.load(reg_off, offset_align);
    let use_stack = bx.icmp(IntPredicate::IntSGE, reg_off_v, zero);
    bx.cond_br(use_stack, &on_stack.llbb(), &maybe_reg.llbb());

    // The value at this point might be in a register, but there is a chance that
    // it could be on the stack so we have to update the offset and then check
    // the offset again.

    if gr_type && layout.align.abi.bytes() > 8 {
        reg_off_v = maybe_reg.add(reg_off_v, bx.const_i32(15));
        reg_off_v = maybe_reg.and(reg_off_v, bx.const_i32(-16));
    }
    let new_reg_off_v = maybe_reg.add(reg_off_v, bx.const_i32(slot_size as i32));

    maybe_reg.store(new_reg_off_v, reg_off, offset_align);

    // Check to see if we have overflowed the registers as a result of this.
    // If we have then we need to use the stack for this value
    let use_stack = maybe_reg.icmp(IntPredicate::IntSGT, new_reg_off_v, zero);
    maybe_reg.cond_br(use_stack, &on_stack.llbb(), &in_reg.llbb());

    let top = in_reg.struct_gep(va_list_addr, reg_top_index);
    let top = in_reg.load(top, bx.tcx().data_layout.pointer_align.abi);

    // reg_value = *(@top + reg_off_v);
    let top = in_reg.gep(top, &[reg_off_v]);
    let top = in_reg.bitcast(top, bx.cx.type_ptr_to(layout.llvm_type(bx)));
    let reg_value = in_reg.load(top, layout.align.abi);
    in_reg.br(&end.llbb());

    // On Stack block
    let stack_value =
        emit_ptr_va_arg(&mut on_stack, list, target_ty, false, Align::from_bytes(8).unwrap(), true);
    on_stack.br(&end.llbb());

    let val = end.phi(
        layout.immediate_llvm_type(bx),
        &[reg_value, stack_value],
        &[&in_reg.llbb(), &on_stack.llbb()],
    );

    *bx = end;
    val
}

pub(super) fn emit_va_arg(
    bx: &mut Builder<'a, 'll, 'tcx>,
    addr: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    // Determine the va_arg implementation to use. The LLVM va_arg instruction
    // is lacking in some instances, so we should only use it as a fallback.
    let target = &bx.cx.tcx.sess.target;
    let arch = &bx.cx.tcx.sess.target.arch;
    match &**arch {
        // Windows x86
        "x86" if target.options.is_like_windows => {
            emit_ptr_va_arg(bx, addr, target_ty, false, Align::from_bytes(4).unwrap(), false)
        }
        // Generic x86
        "x86" => emit_ptr_va_arg(bx, addr, target_ty, false, Align::from_bytes(4).unwrap(), true),
        // Windows AArch64
        "aarch64" if target.options.is_like_windows => {
            emit_ptr_va_arg(bx, addr, target_ty, false, Align::from_bytes(8).unwrap(), false)
        }
        // macOS / iOS AArch64
        "aarch64" if target.options.is_like_osx => {
            emit_ptr_va_arg(bx, addr, target_ty, false, Align::from_bytes(8).unwrap(), true)
        }
        "aarch64" => emit_aapcs_va_arg(bx, addr, target_ty),
        // Windows x86_64
        "x86_64" if target.options.is_like_windows => {
            let target_ty_size = bx.cx.size_of(target_ty).bytes();
            let indirect: bool = target_ty_size > 8 || !target_ty_size.is_power_of_two();
            emit_ptr_va_arg(bx, addr, target_ty, indirect, Align::from_bytes(8).unwrap(), false)
        }
        // For all other architecture/OS combinations fall back to using
        // the LLVM va_arg instruction.
        // https://llvm.org/docs/LangRef.html#va-arg-instruction
        _ => bx.va_arg(addr.immediate(), bx.cx.layout_of(target_ty).llvm_type(bx.cx)),
    }
}
