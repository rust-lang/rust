use crate::builder::Builder;
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::traits::{BaseTypeMethods, BuilderMethods, ConstMethods, DerivedTypeMethods};
use rustc::ty::layout::{Align, HasDataLayout, HasTyCtxt, LayoutOf, Size};
use rustc::ty::Ty;

#[allow(dead_code)]
fn round_pointer_up_to_alignment(
    bx: &mut Builder<'a, 'll, 'tcx>,
    addr: &'ll Value,
    align: Align,
    ptr_ty: &'ll Type
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
    allow_higher_align: bool
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

    if size.bytes() < slot_size.bytes() &&
            &*bx.tcx().sess.target.target.target_endian == "big" {
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
    allow_higher_align: bool
) -> &'ll Value {
    let layout = bx.cx.layout_of(target_ty);
    let (llty, size, align) = if indirect {
        (bx.cx.layout_of(bx.cx.tcx.mk_imm_ptr(target_ty)).llvm_type(bx.cx),
         bx.cx.data_layout().pointer_size,
         bx.cx.data_layout().pointer_align)
    } else {
        (layout.llvm_type(bx.cx),
         layout.size,
         layout.align)
    };
    let (addr, addr_align) = emit_direct_ptr_va_arg(bx, list, llty, size, align.abi,
                                                    slot_size, allow_higher_align);
    if indirect {
        let tmp_ret = bx.load(addr, addr_align);
        bx.load(tmp_ret, align.abi)
    } else {
        bx.load(addr, addr_align)
    }
}

pub(super) fn emit_va_arg(
    bx: &mut Builder<'a, 'll, 'tcx>,
    addr: OperandRef<'tcx, &'ll Value>,
    target_ty: Ty<'tcx>,
) -> &'ll Value {
    // Determine the va_arg implementation to use. The LLVM va_arg instruction
    // is lacking in some instances, so we should only use it as a fallback.
    let target = &bx.cx.tcx.sess.target.target;
    let arch = &bx.cx.tcx.sess.target.target.arch;
    match (&**arch, target.options.is_like_windows) {
        // Windows x86
        ("x86", true) => {
            emit_ptr_va_arg(bx, addr, target_ty, false,
                            Align::from_bytes(4).unwrap(), false)
        }
        // Generic x86
        ("x86", _) => {
            emit_ptr_va_arg(bx, addr, target_ty, false,
                            Align::from_bytes(4).unwrap(), true)
        }
        // Windows AArch64
        ("aarch64", true) => {
            emit_ptr_va_arg(bx, addr, target_ty, false,
                            Align::from_bytes(8).unwrap(), false)
        }
        // iOS AArch64
        ("aarch64", _) if target.target_os == "ios" => {
            emit_ptr_va_arg(bx, addr, target_ty, false,
                            Align::from_bytes(8).unwrap(), true)
        }
        // Windows x86_64
        ("x86_64", true) => {
            let target_ty_size = bx.cx.size_of(target_ty).bytes();
            let indirect = if target_ty_size > 8 || !target_ty_size.is_power_of_two() {
                true
            } else {
                false
            };
            emit_ptr_va_arg(bx, addr, target_ty, indirect,
                            Align::from_bytes(8).unwrap(), false)
        }
        // For all other architecture/OS combinations fall back to using
        // the LLVM va_arg instruction.
        // https://llvm.org/docs/LangRef.html#va-arg-instruction
        _ => bx.va_arg(addr.immediate(), bx.cx.layout_of(target_ty).llvm_type(bx.cx))
    }
}
