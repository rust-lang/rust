//! Codegen of `asm!` invocations.

use std::fmt::Write;

use cranelift_codegen::isa::CallConv;
use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::LangItem;
use rustc_span::sym;
use rustc_target::asm::*;
use target_lexicon::BinaryFormat;

use crate::prelude::*;

pub(crate) enum CInlineAsmOperand<'tcx> {
    In {
        reg: InlineAsmRegOrRegClass,
        value: Value,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        place: Option<CPlace<'tcx>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        _late: bool,
        in_value: Value,
        out_place: Option<CPlace<'tcx>>,
    },
    Const {
        value: String,
    },
    Symbol {
        symbol: String,
    },
}

pub(crate) fn codegen_inline_asm_terminator<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    span: Span,
    template: &[InlineAsmTemplatePiece],
    operands: &[InlineAsmOperand<'tcx>],
    options: InlineAsmOptions,
    destination: Option<mir::BasicBlock>,
) {
    // Used by panic_abort on Windows, but uses a syntax which only happens to work with
    // asm!() by accident and breaks with the GNU assembler as well as global_asm!() for
    // the LLVM backend.
    if template.len() == 1 && template[0] == InlineAsmTemplatePiece::String("int $$0x29".into()) {
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return;
    }

    let operands = operands
        .iter()
        .map(|operand| match *operand {
            InlineAsmOperand::In { reg, ref value } => CInlineAsmOperand::In {
                reg,
                value: crate::base::codegen_operand(fx, value).load_scalar(fx),
            },
            InlineAsmOperand::Out { reg, late, ref place } => CInlineAsmOperand::Out {
                reg,
                late,
                place: place.map(|place| crate::base::codegen_place(fx, place)),
            },
            InlineAsmOperand::InOut { reg, late, ref in_value, ref out_place } => {
                CInlineAsmOperand::InOut {
                    reg,
                    _late: late,
                    in_value: crate::base::codegen_operand(fx, in_value).load_scalar(fx),
                    out_place: out_place.map(|place| crate::base::codegen_place(fx, place)),
                }
            }
            InlineAsmOperand::Const { ref value } => {
                let (const_value, ty) = crate::constant::eval_mir_constant(fx, value);
                let value = rustc_codegen_ssa::common::asm_const_to_str(
                    fx.tcx,
                    span,
                    const_value,
                    fx.layout_of(ty),
                );
                CInlineAsmOperand::Const { value }
            }
            InlineAsmOperand::SymFn { ref value } => {
                if cfg!(not(feature = "inline_asm_sym")) {
                    fx.tcx
                        .dcx()
                        .span_err(span, "asm! and global_asm! sym operands are not yet supported");
                }

                let const_ = fx.monomorphize(value.const_);
                if let ty::FnDef(def_id, args) = *const_.ty().kind() {
                    let instance = ty::Instance::resolve_for_fn_ptr(
                        fx.tcx,
                        ty::TypingEnv::fully_monomorphized(),
                        def_id,
                        args,
                    )
                    .unwrap();
                    let symbol = fx.tcx.symbol_name(instance);

                    // Pass a wrapper rather than the function itself as the function itself may not
                    // be exported from the main codegen unit and may thus be unreachable from the
                    // object file created by an external assembler.
                    let wrapper_name = format!(
                        "__inline_asm_{}_wrapper_n{}",
                        fx.cx.cgu_name.as_str().replace('.', "__").replace('-', "_"),
                        fx.cx.inline_asm_index
                    );
                    fx.cx.inline_asm_index += 1;
                    let sig =
                        get_function_sig(fx.tcx, fx.target_config.default_call_conv, instance);
                    create_wrapper_function(fx.module, sig, &wrapper_name, symbol.name);

                    CInlineAsmOperand::Symbol { symbol: wrapper_name }
                } else {
                    span_bug!(span, "invalid type for asm sym (fn)");
                }
            }
            InlineAsmOperand::SymStatic { def_id } => {
                assert!(fx.tcx.is_static(def_id));
                let instance = Instance::mono(fx.tcx, def_id);
                CInlineAsmOperand::Symbol { symbol: fx.tcx.symbol_name(instance).name.to_owned() }
            }
            InlineAsmOperand::Label { .. } => {
                span_bug!(span, "asm! label operands are not yet supported");
            }
        })
        .collect::<Vec<_>>();

    codegen_inline_asm_inner(fx, template, &operands, options);

    match destination {
        Some(destination) => {
            let destination_block = fx.get_block(destination);
            fx.bcx.ins().jump(destination_block, &[]);
        }
        None => {
            fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
        }
    }
}

pub(crate) fn codegen_inline_asm_inner<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    template: &[InlineAsmTemplatePiece],
    operands: &[CInlineAsmOperand<'tcx>],
    options: InlineAsmOptions,
) {
    // FIXME add .eh_frame unwind info directives

    let mut asm_gen = InlineAssemblyGenerator {
        tcx: fx.tcx,
        arch: fx.tcx.sess.asm_arch.unwrap(),
        enclosing_def_id: fx.instance.def_id(),
        template,
        operands,
        options,
        registers: Vec::new(),
        stack_slots_clobber: Vec::new(),
        stack_slots_input: Vec::new(),
        stack_slots_output: Vec::new(),
        stack_slot_size: Size::from_bytes(0),
    };
    asm_gen.allocate_registers();
    asm_gen.allocate_stack_slots();

    let asm_name = format!(
        "__inline_asm_{}_n{}",
        fx.cx.cgu_name.as_str().replace('.', "__").replace('-', "_"),
        fx.cx.inline_asm_index
    );
    fx.cx.inline_asm_index += 1;

    let generated_asm = asm_gen.generate_asm_wrapper(&asm_name);
    fx.cx.global_asm.push_str(&generated_asm);

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (i, operand) in operands.iter().enumerate() {
        match operand {
            CInlineAsmOperand::In { reg: _, value } => {
                inputs.push((asm_gen.stack_slots_input[i].unwrap(), *value));
            }
            CInlineAsmOperand::Out { reg: _, late: _, place } => {
                if let Some(place) = place {
                    outputs.push((asm_gen.stack_slots_output[i].unwrap(), *place));
                }
            }
            CInlineAsmOperand::InOut { reg: _, _late: _, in_value, out_place } => {
                inputs.push((asm_gen.stack_slots_input[i].unwrap(), *in_value));
                if let Some(out_place) = out_place {
                    outputs.push((asm_gen.stack_slots_output[i].unwrap(), *out_place));
                }
            }
            CInlineAsmOperand::Const { value: _ } | CInlineAsmOperand::Symbol { symbol: _ } => {}
        }
    }

    call_inline_asm(fx, &asm_name, asm_gen.stack_slot_size, inputs, outputs);
}

struct InlineAssemblyGenerator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    arch: InlineAsmArch,
    enclosing_def_id: DefId,
    template: &'a [InlineAsmTemplatePiece],
    operands: &'a [CInlineAsmOperand<'tcx>],
    options: InlineAsmOptions,
    registers: Vec<Option<InlineAsmReg>>,
    stack_slots_clobber: Vec<Option<Size>>,
    stack_slots_input: Vec<Option<Size>>,
    stack_slots_output: Vec<Option<Size>>,
    stack_slot_size: Size,
}

impl<'tcx> InlineAssemblyGenerator<'_, 'tcx> {
    fn allocate_registers(&mut self) {
        let sess = self.tcx.sess;
        let map = allocatable_registers(
            self.arch,
            sess.relocation_model(),
            self.tcx.asm_target_features(self.enclosing_def_id),
            &sess.target,
        );
        let mut allocated = FxHashMap::<_, (bool, bool)>::default();
        let mut regs = vec![None; self.operands.len()];

        // Add explicit registers to the allocated set.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                CInlineAsmOperand::In { reg: InlineAsmRegOrRegClass::Reg(reg), .. } => {
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().0 = true;
                }
                CInlineAsmOperand::Out {
                    reg: InlineAsmRegOrRegClass::Reg(reg),
                    late: true,
                    ..
                } => {
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().1 = true;
                }
                CInlineAsmOperand::Out { reg: InlineAsmRegOrRegClass::Reg(reg), .. }
                | CInlineAsmOperand::InOut { reg: InlineAsmRegOrRegClass::Reg(reg), .. } => {
                    regs[i] = Some(reg);
                    allocated.insert(reg, (true, true));
                }
                _ => (),
            }
        }

        // Allocate out/inout/inlateout registers first because they are more constrained.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                CInlineAsmOperand::Out {
                    reg: InlineAsmRegOrRegClass::RegClass(class),
                    late: false,
                    ..
                }
                | CInlineAsmOperand::InOut {
                    reg: InlineAsmRegOrRegClass::RegClass(class), ..
                } => {
                    let mut alloc_reg = None;
                    for &reg in &map[&class] {
                        let mut used = false;
                        reg.overlapping_regs(|r| {
                            if allocated.contains_key(&r) {
                                used = true;
                            }
                        });

                        if !used {
                            alloc_reg = Some(reg);
                            break;
                        }
                    }

                    let reg = alloc_reg.expect("cannot allocate registers");
                    regs[i] = Some(reg);
                    allocated.insert(reg, (true, true));
                }
                _ => (),
            }
        }

        // Allocate in/lateout.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                CInlineAsmOperand::In { reg: InlineAsmRegOrRegClass::RegClass(class), .. } => {
                    let mut alloc_reg = None;
                    for &reg in &map[&class] {
                        let mut used = false;
                        reg.overlapping_regs(|r| {
                            if allocated.get(&r).copied().unwrap_or_default().0 {
                                used = true;
                            }
                        });

                        if !used {
                            alloc_reg = Some(reg);
                            break;
                        }
                    }

                    let reg = alloc_reg.expect("cannot allocate registers");
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().0 = true;
                }
                CInlineAsmOperand::Out {
                    reg: InlineAsmRegOrRegClass::RegClass(class),
                    late: true,
                    ..
                } => {
                    let mut alloc_reg = None;
                    for &reg in &map[&class] {
                        let mut used = false;
                        reg.overlapping_regs(|r| {
                            if allocated.get(&r).copied().unwrap_or_default().1 {
                                used = true;
                            }
                        });

                        if !used {
                            alloc_reg = Some(reg);
                            break;
                        }
                    }

                    let reg = alloc_reg.expect("cannot allocate registers");
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().1 = true;
                }
                _ => (),
            }
        }

        self.registers = regs;
    }

    fn allocate_stack_slots(&mut self) {
        let mut slot_size = Size::from_bytes(0);
        let mut slots_clobber = vec![None; self.operands.len()];
        let mut slots_input = vec![None; self.operands.len()];
        let mut slots_output = vec![None; self.operands.len()];

        let new_slot_fn = |slot_size: &mut Size, reg_class: InlineAsmRegClass| {
            let reg_size = reg_class
                .supported_types(self.arch, true)
                .iter()
                .map(|(ty, _)| ty.size())
                .max()
                .unwrap();
            let align = rustc_abi::Align::from_bytes(reg_size.bytes()).unwrap();
            let offset = slot_size.align_to(align);
            *slot_size = offset + reg_size;
            offset
        };
        let mut new_slot = |x| new_slot_fn(&mut slot_size, x);

        // Allocate stack slots for saving clobbered registers
        let abi_clobber = InlineAsmClobberAbi::parse(
            self.arch,
            &self.tcx.sess.target,
            &self.tcx.sess.unstable_target_features,
            sym::C,
        )
        .unwrap()
        .clobbered_regs();
        for (i, reg) in self.registers.iter().enumerate().filter_map(|(i, r)| r.map(|r| (i, r))) {
            let mut need_save = true;
            // If the register overlaps with a register clobbered by function call, then
            // we don't need to save it.
            for r in abi_clobber {
                r.overlapping_regs(|r| {
                    if r == reg {
                        need_save = false;
                    }
                });

                if !need_save {
                    break;
                }
            }

            if need_save {
                slots_clobber[i] = Some(new_slot(reg.reg_class()));
            }
        }

        // Allocate stack slots for inout
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                CInlineAsmOperand::InOut { reg, out_place: Some(_), .. } => {
                    let slot = new_slot(reg.reg_class());
                    slots_input[i] = Some(slot);
                    slots_output[i] = Some(slot);
                }
                _ => (),
            }
        }

        let slot_size_before_input = slot_size;
        let mut new_slot = |x| new_slot_fn(&mut slot_size, x);

        // Allocate stack slots for input
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                CInlineAsmOperand::In { reg, .. }
                | CInlineAsmOperand::InOut { reg, out_place: None, .. } => {
                    slots_input[i] = Some(new_slot(reg.reg_class()));
                }
                _ => (),
            }
        }

        // Reset slot size to before input so that input and output operands can overlap
        // and save some memory.
        let slot_size_after_input = slot_size;
        slot_size = slot_size_before_input;
        let mut new_slot = |x| new_slot_fn(&mut slot_size, x);

        // Allocate stack slots for output
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                CInlineAsmOperand::Out { reg, place: Some(_), .. } => {
                    slots_output[i] = Some(new_slot(reg.reg_class()));
                }
                _ => (),
            }
        }

        slot_size = slot_size.max(slot_size_after_input);

        self.stack_slots_clobber = slots_clobber;
        self.stack_slots_input = slots_input;
        self.stack_slots_output = slots_output;
        self.stack_slot_size = slot_size;
    }

    fn generate_asm_wrapper(&self, asm_name: &str) -> String {
        let binary_format = crate::target_triple(self.tcx.sess).binary_format;

        let mut generated_asm = String::new();
        match binary_format {
            BinaryFormat::Elf => {
                writeln!(generated_asm, ".globl {}", asm_name).unwrap();
                writeln!(generated_asm, ".type {},@function", asm_name).unwrap();
                writeln!(generated_asm, ".section .text.{},\"ax\",@progbits", asm_name).unwrap();
                writeln!(generated_asm, "{}:", asm_name).unwrap();
            }
            BinaryFormat::Macho => {
                writeln!(generated_asm, ".globl _{}", asm_name).unwrap();
                writeln!(generated_asm, "_{}:", asm_name).unwrap();
            }
            BinaryFormat::Coff => {
                writeln!(generated_asm, ".globl {}", asm_name).unwrap();
                writeln!(generated_asm, "{}:", asm_name).unwrap();
            }
            _ => self
                .tcx
                .dcx()
                .fatal(format!("Unsupported binary format for inline asm: {binary_format:?}")),
        }

        let is_x86 = matches!(self.arch, InlineAsmArch::X86 | InlineAsmArch::X86_64);

        if is_x86 {
            generated_asm.push_str(".intel_syntax noprefix\n");
        }

        Self::prologue(&mut generated_asm, self.arch);

        // Save clobbered registers
        if !self.options.contains(InlineAsmOptions::NORETURN) {
            for (reg, slot) in self
                .registers
                .iter()
                .zip(self.stack_slots_clobber.iter().copied())
                .filter_map(|(r, s)| r.zip(s))
            {
                Self::save_register(&mut generated_asm, self.arch, reg, slot);
            }
        }

        // Write input registers
        for (reg, slot) in self
            .registers
            .iter()
            .zip(self.stack_slots_input.iter().copied())
            .filter_map(|(r, s)| r.zip(s))
        {
            Self::restore_register(&mut generated_asm, self.arch, reg, slot);
        }

        if is_x86 && self.options.contains(InlineAsmOptions::ATT_SYNTAX) {
            generated_asm.push_str(".att_syntax\n");
        }

        if self.arch == InlineAsmArch::AArch64 {
            for feature in &self.tcx.codegen_fn_attrs(self.enclosing_def_id).target_features {
                if feature.name == sym::neon {
                    continue;
                }
                writeln!(generated_asm, ".arch_extension {}", feature.name).unwrap();
            }
        }

        // The actual inline asm
        for piece in self.template {
            match piece {
                InlineAsmTemplatePiece::String(s) => {
                    generated_asm.push_str(s);
                }
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier, span: _ } => {
                    match self.operands[*operand_idx] {
                        CInlineAsmOperand::In { .. }
                        | CInlineAsmOperand::Out { .. }
                        | CInlineAsmOperand::InOut { .. } => {
                            if self.options.contains(InlineAsmOptions::ATT_SYNTAX) {
                                generated_asm.push('%');
                            }

                            let reg = self.registers[*operand_idx].unwrap();
                            match self.arch {
                                InlineAsmArch::X86_64 => match reg {
                                    InlineAsmReg::X86(reg)
                                        if reg as u32 >= X86InlineAsmReg::xmm0 as u32
                                            && reg as u32 <= X86InlineAsmReg::xmm15 as u32 =>
                                    {
                                        // rustc emits x0 rather than xmm0
                                        let class = match *modifier {
                                            None | Some('x') => "xmm",
                                            Some('y') => "ymm",
                                            Some('z') => "zmm",
                                            _ => unreachable!(),
                                        };
                                        write!(
                                            generated_asm,
                                            "{class}{}",
                                            reg as u32 - X86InlineAsmReg::xmm0 as u32
                                        )
                                        .unwrap();
                                    }
                                    _ => reg
                                        .emit(&mut generated_asm, InlineAsmArch::X86_64, *modifier)
                                        .unwrap(),
                                },
                                InlineAsmArch::AArch64 => match reg {
                                    InlineAsmReg::AArch64(reg) if reg.vreg_index().is_some() => {
                                        // rustc emits v0 rather than q0
                                        reg.emit(
                                            &mut generated_asm,
                                            InlineAsmArch::AArch64,
                                            Some(modifier.unwrap_or('q')),
                                        )
                                        .unwrap()
                                    }
                                    _ => reg
                                        .emit(&mut generated_asm, InlineAsmArch::AArch64, *modifier)
                                        .unwrap(),
                                },
                                _ => reg.emit(&mut generated_asm, self.arch, *modifier).unwrap(),
                            }
                        }
                        CInlineAsmOperand::Const { ref value } => {
                            generated_asm.push_str(value);
                        }
                        CInlineAsmOperand::Symbol { ref symbol } => generated_asm.push_str(symbol),
                    }
                }
            }
        }
        generated_asm.push('\n');

        if self.arch == InlineAsmArch::AArch64 {
            for feature in &self.tcx.codegen_fn_attrs(self.enclosing_def_id).target_features {
                if feature.name == sym::neon {
                    continue;
                }
                writeln!(generated_asm, ".arch_extension no{}", feature.name).unwrap();
            }
        }

        if is_x86 && self.options.contains(InlineAsmOptions::ATT_SYNTAX) {
            generated_asm.push_str(".intel_syntax noprefix\n");
        }

        if !self.options.contains(InlineAsmOptions::NORETURN) {
            // Read output registers
            for (reg, slot) in self
                .registers
                .iter()
                .zip(self.stack_slots_output.iter().copied())
                .filter_map(|(r, s)| r.zip(s))
            {
                Self::save_register(&mut generated_asm, self.arch, reg, slot);
            }

            // Restore clobbered registers
            for (reg, slot) in self
                .registers
                .iter()
                .zip(self.stack_slots_clobber.iter().copied())
                .filter_map(|(r, s)| r.zip(s))
            {
                Self::restore_register(&mut generated_asm, self.arch, reg, slot);
            }

            Self::epilogue(&mut generated_asm, self.arch);
        } else {
            Self::epilogue_noreturn(&mut generated_asm, self.arch);
        }

        if is_x86 {
            generated_asm.push_str(".att_syntax\n");
        }

        match binary_format {
            BinaryFormat::Elf => {
                writeln!(generated_asm, ".size {name}, .-{name}", name = asm_name).unwrap();
                generated_asm.push_str(".text\n");
            }
            BinaryFormat::Macho | BinaryFormat::Coff => {}
            _ => self
                .tcx
                .dcx()
                .fatal(format!("Unsupported binary format for inline asm: {binary_format:?}")),
        }

        generated_asm.push_str("\n\n");

        generated_asm
    }

    fn prologue(generated_asm: &mut String, arch: InlineAsmArch) {
        match arch {
            InlineAsmArch::X86_64 => {
                generated_asm.push_str("    push rbp\n");
                generated_asm.push_str("    mov rbp,rsp\n");
                generated_asm.push_str("    push rbx\n"); // rbx is callee saved
                // rbx is reserved by LLVM for the "base pointer", so rustc doesn't allow using it
                generated_asm.push_str("    mov rbx,rdi\n");
            }
            InlineAsmArch::AArch64 => {
                generated_asm.push_str("    stp fp, lr, [sp, #-32]!\n");
                generated_asm.push_str("    mov fp, sp\n");
                generated_asm.push_str("    str x19, [sp, #24]\n"); // x19 is callee saved
                // x19 is reserved by LLVM for the "base pointer", so rustc doesn't allow using it
                generated_asm.push_str("    mov x19, x0\n");
            }
            InlineAsmArch::RiscV64 => {
                generated_asm.push_str("    addi sp, sp, -16\n");
                generated_asm.push_str("    sd ra, 8(sp)\n");
                generated_asm.push_str("    sd s1, 0(sp)\n"); // s1 is callee saved
                // s1/x9 is reserved by LLVM for the "base pointer", so rustc doesn't allow using it
                generated_asm.push_str("    mv s1, a0\n");
            }
            _ => unimplemented!("prologue for {:?}", arch),
        }
    }

    fn epilogue(generated_asm: &mut String, arch: InlineAsmArch) {
        match arch {
            InlineAsmArch::X86_64 => {
                generated_asm.push_str("    pop rbx\n");
                generated_asm.push_str("    pop rbp\n");
                generated_asm.push_str("    ret\n");
            }
            InlineAsmArch::AArch64 => {
                generated_asm.push_str("    ldr x19, [sp, #24]\n");
                generated_asm.push_str("    ldp fp, lr, [sp], #32\n");
                generated_asm.push_str("    ret\n");
            }
            InlineAsmArch::RiscV64 => {
                generated_asm.push_str("    ld s1, 0(sp)\n");
                generated_asm.push_str("    ld ra, 8(sp)\n");
                generated_asm.push_str("    addi sp, sp, 16\n");
                generated_asm.push_str("    ret\n");
            }
            _ => unimplemented!("epilogue for {:?}", arch),
        }
    }

    fn epilogue_noreturn(generated_asm: &mut String, arch: InlineAsmArch) {
        match arch {
            InlineAsmArch::X86_64 => {
                generated_asm.push_str("    ud2\n");
            }
            InlineAsmArch::AArch64 => {
                generated_asm.push_str("    brk     #0x1\n");
            }
            InlineAsmArch::RiscV64 => {
                generated_asm.push_str("    ebreak\n");
            }
            _ => unimplemented!("epilogue_noreturn for {:?}", arch),
        }
    }

    fn save_register(
        generated_asm: &mut String,
        arch: InlineAsmArch,
        reg: InlineAsmReg,
        offset: Size,
    ) {
        match arch {
            InlineAsmArch::X86_64 => {
                match reg {
                    InlineAsmReg::X86(reg)
                        if reg as u32 >= X86InlineAsmReg::xmm0 as u32
                            && reg as u32 <= X86InlineAsmReg::xmm15 as u32 =>
                    {
                        // rustc emits x0 rather than xmm0
                        write!(generated_asm, "    movups [rbx+0x{:x}], ", offset.bytes()).unwrap();
                        write!(generated_asm, "xmm{}", reg as u32 - X86InlineAsmReg::xmm0 as u32)
                            .unwrap();
                    }
                    _ => {
                        write!(generated_asm, "    mov [rbx+0x{:x}], ", offset.bytes()).unwrap();
                        reg.emit(generated_asm, InlineAsmArch::X86_64, None).unwrap();
                    }
                }
                generated_asm.push('\n');
            }
            InlineAsmArch::AArch64 => {
                generated_asm.push_str("    str ");
                match reg {
                    InlineAsmReg::AArch64(reg) if reg.vreg_index().is_some() => {
                        // rustc emits v0 rather than q0
                        reg.emit(generated_asm, InlineAsmArch::AArch64, Some('q')).unwrap()
                    }
                    _ => reg.emit(generated_asm, InlineAsmArch::AArch64, None).unwrap(),
                }
                writeln!(generated_asm, ", [x19, 0x{:x}]", offset.bytes()).unwrap();
            }
            InlineAsmArch::RiscV64 => {
                generated_asm.push_str("    sd ");
                reg.emit(generated_asm, InlineAsmArch::RiscV64, None).unwrap();
                writeln!(generated_asm, ", 0x{:x}(s1)", offset.bytes()).unwrap();
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
                match reg {
                    InlineAsmReg::X86(reg)
                        if reg as u32 >= X86InlineAsmReg::xmm0 as u32
                            && reg as u32 <= X86InlineAsmReg::xmm15 as u32 =>
                    {
                        // rustc emits x0 rather than xmm0
                        write!(
                            generated_asm,
                            "    movups xmm{}",
                            reg as u32 - X86InlineAsmReg::xmm0 as u32
                        )
                        .unwrap();
                    }
                    _ => {
                        generated_asm.push_str("    mov ");
                        reg.emit(generated_asm, InlineAsmArch::X86_64, None).unwrap()
                    }
                }
                writeln!(generated_asm, ", [rbx+0x{:x}]", offset.bytes()).unwrap();
            }
            InlineAsmArch::AArch64 => {
                generated_asm.push_str("    ldr ");
                match reg {
                    InlineAsmReg::AArch64(reg) if reg.vreg_index().is_some() => {
                        // rustc emits v0 rather than q0
                        reg.emit(generated_asm, InlineAsmArch::AArch64, Some('q')).unwrap()
                    }
                    _ => reg.emit(generated_asm, InlineAsmArch::AArch64, None).unwrap(),
                }
                writeln!(generated_asm, ", [x19, 0x{:x}]", offset.bytes()).unwrap();
            }
            InlineAsmArch::RiscV64 => {
                generated_asm.push_str("    ld ");
                reg.emit(generated_asm, InlineAsmArch::RiscV64, None).unwrap();
                writeln!(generated_asm, ", 0x{:x}(s1)", offset.bytes()).unwrap();
            }
            _ => unimplemented!("restore_register for {:?}", arch),
        }
    }
}

fn call_inline_asm<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    asm_name: &str,
    slot_size: Size,
    inputs: Vec<(Size, Value)>,
    outputs: Vec<(Size, CPlace<'tcx>)>,
) {
    let stack_slot =
        fx.create_stack_slot(u32::try_from(slot_size.bytes().next_multiple_of(16)).unwrap(), 16);

    let inline_asm_func = fx
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
    let inline_asm_func = fx.module.declare_func_in_func(inline_asm_func, fx.bcx.func);
    if fx.clif_comments.enabled() {
        fx.add_comment(inline_asm_func, asm_name);
    }

    for (offset, value) in inputs {
        stack_slot.offset(fx, i32::try_from(offset.bytes()).unwrap().into()).store(
            fx,
            value,
            MemFlags::trusted(),
        );
    }

    let stack_slot_addr = stack_slot.get_addr(fx);
    fx.bcx.ins().call(inline_asm_func, &[stack_slot_addr]);

    for (offset, place) in outputs {
        let ty = if place.layout().ty.is_simd() {
            let (lane_count, lane_type) = place.layout().ty.simd_size_and_type(fx.tcx);
            asm_clif_type(fx, lane_type).unwrap().by(lane_count.try_into().unwrap()).unwrap()
        } else {
            asm_clif_type(fx, place.layout().ty).unwrap()
        };
        let value = stack_slot.offset(fx, i32::try_from(offset.bytes()).unwrap().into()).load(
            fx,
            ty,
            MemFlags::trusted(),
        );
        place.write_cvalue(fx, CValue::by_val(value, place.layout()));
    }
}

fn asm_clif_type<'tcx>(fx: &FunctionCx<'_, '_, 'tcx>, ty: Ty<'tcx>) -> Option<types::Type> {
    match ty.kind() {
        // Adapted from https://github.com/rust-lang/rust/blob/f3c66088610c1b80110297c2d9a8b5f9265b013f/compiler/rustc_hir_analysis/src/check/intrinsicck.rs#L136-L151
        ty::Adt(adt, args) if fx.tcx.is_lang_item(adt.did(), LangItem::MaybeUninit) => {
            let fields = &adt.non_enum_variant().fields;
            let ty = fields[FieldIdx::ONE].ty(fx.tcx, args);
            let ty::Adt(ty, args) = ty.kind() else {
                unreachable!("expected first field of `MaybeUninit` to be an ADT")
            };
            assert!(
                ty.is_manually_drop(),
                "expected first field of `MaybeUninit` to be `ManuallyDrop`"
            );
            let fields = &ty.non_enum_variant().fields;
            let ty = fields[FieldIdx::ZERO].ty(fx.tcx, args);
            fx.clif_type(ty)
        }
        _ => fx.clif_type(ty),
    }
}
