//! Codegen of [`asm!`] invocations.

use crate::prelude::*;

use std::fmt::Write;

use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_middle::mir::InlineAsmOperand;
use rustc_target::asm::*;

pub(crate) fn codegen_inline_asm<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    _span: Span,
    template: &[InlineAsmTemplatePiece],
    operands: &[InlineAsmOperand<'tcx>],
    options: InlineAsmOptions,
) {
    // FIXME add .eh_frame unwind info directives

    if template.is_empty() {
        // Black box
        return;
    }

    let mut slot_size = Size::from_bytes(0);
    let mut clobbered_regs = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    let mut new_slot = |reg_class: InlineAsmRegClass| {
        let reg_size = reg_class
            .supported_types(InlineAsmArch::X86_64)
            .iter()
            .map(|(ty, _)| ty.size())
            .max()
            .unwrap();
        let align = rustc_target::abi::Align::from_bytes(reg_size.bytes()).unwrap();
        slot_size = slot_size.align_to(align);
        let offset = slot_size;
        slot_size += reg_size;
        offset
    };

    // FIXME overlap input and output slots to save stack space
    for operand in operands {
        match *operand {
            InlineAsmOperand::In { reg, ref value } => {
                let reg = expect_reg(reg);
                clobbered_regs.push((reg, new_slot(reg.reg_class())));
                inputs.push((
                    reg,
                    new_slot(reg.reg_class()),
                    crate::base::codegen_operand(fx, value).load_scalar(fx),
                ));
            }
            InlineAsmOperand::Out {
                reg,
                late: _,
                place,
            } => {
                let reg = expect_reg(reg);
                clobbered_regs.push((reg, new_slot(reg.reg_class())));
                if let Some(place) = place {
                    outputs.push((
                        reg,
                        new_slot(reg.reg_class()),
                        crate::base::codegen_place(fx, place),
                    ));
                }
            }
            InlineAsmOperand::InOut {
                reg,
                late: _,
                ref in_value,
                out_place,
            } => {
                let reg = expect_reg(reg);
                clobbered_regs.push((reg, new_slot(reg.reg_class())));
                inputs.push((
                    reg,
                    new_slot(reg.reg_class()),
                    crate::base::codegen_operand(fx, in_value).load_scalar(fx),
                ));
                if let Some(out_place) = out_place {
                    outputs.push((
                        reg,
                        new_slot(reg.reg_class()),
                        crate::base::codegen_place(fx, out_place),
                    ));
                }
            }
            InlineAsmOperand::Const { value: _ } => todo!(),
            InlineAsmOperand::SymFn { value: _ } => todo!(),
            InlineAsmOperand::SymStatic { def_id: _ } => todo!(),
        }
    }

    let inline_asm_index = fx.inline_asm_index;
    fx.inline_asm_index += 1;
    let asm_name = format!(
        "{}__inline_asm_{}",
        fx.tcx.symbol_name(fx.instance).name,
        inline_asm_index
    );

    let generated_asm = generate_asm_wrapper(
        &asm_name,
        InlineAsmArch::X86_64,
        options,
        template,
        clobbered_regs,
        &inputs,
        &outputs,
    );
    fx.cx.global_asm.push_str(&generated_asm);

    call_inline_asm(fx, &asm_name, slot_size, inputs, outputs);
}

fn generate_asm_wrapper(
    asm_name: &str,
    arch: InlineAsmArch,
    options: InlineAsmOptions,
    template: &[InlineAsmTemplatePiece],
    clobbered_regs: Vec<(InlineAsmReg, Size)>,
    inputs: &[(InlineAsmReg, Size, Value)],
    outputs: &[(InlineAsmReg, Size, CPlace<'_>)],
) -> String {
    let mut generated_asm = String::new();
    writeln!(generated_asm, ".globl {}", asm_name).unwrap();
    writeln!(generated_asm, ".type {},@function", asm_name).unwrap();
    writeln!(
        generated_asm,
        ".section .text.{},\"ax\",@progbits",
        asm_name
    )
    .unwrap();
    writeln!(generated_asm, "{}:", asm_name).unwrap();

    generated_asm.push_str(".intel_syntax noprefix\n");
    generated_asm.push_str("    push rbp\n");
    generated_asm.push_str("    mov rbp,rdi\n");

    // Save clobbered registers
    if !options.contains(InlineAsmOptions::NORETURN) {
        // FIXME skip registers saved by the calling convention
        for &(reg, offset) in &clobbered_regs {
            save_register(&mut generated_asm, arch, reg, offset);
        }
    }

    // Write input registers
    for &(reg, offset, _value) in inputs {
        restore_register(&mut generated_asm, arch, reg, offset);
    }

    if options.contains(InlineAsmOptions::ATT_SYNTAX) {
        generated_asm.push_str(".att_syntax\n");
    }

    // The actual inline asm
    for piece in template {
        match piece {
            InlineAsmTemplatePiece::String(s) => {
                generated_asm.push_str(s);
            }
            InlineAsmTemplatePiece::Placeholder {
                operand_idx: _,
                modifier: _,
                span: _,
            } => todo!(),
        }
    }
    generated_asm.push('\n');

    if options.contains(InlineAsmOptions::ATT_SYNTAX) {
        generated_asm.push_str(".intel_syntax noprefix\n");
    }

    if !options.contains(InlineAsmOptions::NORETURN) {
        // Read output registers
        for &(reg, offset, _place) in outputs {
            save_register(&mut generated_asm, arch, reg, offset);
        }

        // Restore clobbered registers
        for &(reg, offset) in clobbered_regs.iter().rev() {
            restore_register(&mut generated_asm, arch, reg, offset);
        }

        generated_asm.push_str("    pop rbp\n");
        generated_asm.push_str("    ret\n");
    } else {
        generated_asm.push_str("    ud2\n");
    }

    generated_asm.push_str(".att_syntax\n");
    writeln!(generated_asm, ".size {name}, .-{name}", name = asm_name).unwrap();
    generated_asm.push_str(".text\n");
    generated_asm.push_str("\n\n");

    generated_asm
}

fn call_inline_asm<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    asm_name: &str,
    slot_size: Size,
    inputs: Vec<(InlineAsmReg, Size, Value)>,
    outputs: Vec<(InlineAsmReg, Size, CPlace<'tcx>)>,
) {
    let stack_slot = fx.bcx.func.create_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        offset: None,
        size: u32::try_from(slot_size.bytes()).unwrap(),
    });
    #[cfg(debug_assertions)]
    fx.add_comment(stack_slot, "inline asm scratch slot");

    let inline_asm_func = fx
        .cx
        .module
        .declare_function(
            asm_name,
            Linkage::Import,
            &Signature {
                call_conv: CallConv::SystemV,
                params: vec![AbiParam::new(fx.pointer_type)],
                returns: vec![],
            },
        )
        .unwrap();
    let inline_asm_func = fx
        .cx
        .module
        .declare_func_in_func(inline_asm_func, &mut fx.bcx.func);
    #[cfg(debug_assertions)]
    fx.add_comment(inline_asm_func, asm_name);

    for (_reg, offset, value) in inputs {
        fx.bcx
            .ins()
            .stack_store(value, stack_slot, i32::try_from(offset.bytes()).unwrap());
    }

    let stack_slot_addr = fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, 0);
    fx.bcx.ins().call(inline_asm_func, &[stack_slot_addr]);

    for (_reg, offset, place) in outputs {
        let ty = fx.clif_type(place.layout().ty).unwrap();
        let value = fx
            .bcx
            .ins()
            .stack_load(ty, stack_slot, i32::try_from(offset.bytes()).unwrap());
        place.write_cvalue(fx, CValue::by_val(value, place.layout()));
    }
}

fn expect_reg(reg_or_class: InlineAsmRegOrRegClass) -> InlineAsmReg {
    match reg_or_class {
        InlineAsmRegOrRegClass::Reg(reg) => reg,
        InlineAsmRegOrRegClass::RegClass(class) => unimplemented!("{:?}", class),
    }
}

fn save_register(generated_asm: &mut String, arch: InlineAsmArch, reg: InlineAsmReg, offset: Size) {
    match arch {
        InlineAsmArch::X86_64 => {
            write!(generated_asm, "    mov [rbp+0x{:x}], ", offset.bytes()).unwrap();
            reg.emit(generated_asm, InlineAsmArch::X86_64, None)
                .unwrap();
            generated_asm.push('\n');
        }
        _ => unimplemented!("save_register for {:?}", arch),
    }
}

fn restore_register(
    generated_asm: &mut String,
    arch: InlineAsmArch,
    reg: InlineAsmReg,
    offset: Size,
) {
    match arch {
        InlineAsmArch::X86_64 => {
            generated_asm.push_str("    mov ");
            reg.emit(generated_asm, InlineAsmArch::X86_64, None)
                .unwrap();
            writeln!(generated_asm, ", [rbp+0x{:x}]", offset.bytes()).unwrap();
        }
        _ => unimplemented!("restore_register for {:?}", arch),
    }
}
