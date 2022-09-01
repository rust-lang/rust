use crate::attributes;
use crate::builder::Builder;
use crate::common::Funclet;
use crate::context::CodegenCx;
use crate::llvm;
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::{bug, span_bug, ty::Instance};
use rustc_span::{Pos, Span};
use rustc_target::abi::*;
use rustc_target::asm::*;

use libc::{c_char, c_uint};
use smallvec::SmallVec;

impl<'ll, 'tcx> AsmBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        line_spans: &[Span],
        instance: Instance<'_>,
        dest_catch_funclet: Option<(Self::BasicBlock, Self::BasicBlock, Option<&Self::Funclet>)>,
    ) {
        let asm_arch = self.tcx.sess.asm_arch.unwrap();

        // Collect the types of output operands
        let mut constraints = vec![];
        let mut clobbers = vec![];
        let mut output_types = vec![];
        let mut op_idx = FxHashMap::default();
        let mut clobbered_x87 = false;
        for (idx, op) in operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::Out { reg, late, place } => {
                    let is_target_supported = |reg_class: InlineAsmRegClass| {
                        for &(_, feature) in reg_class.supported_types(asm_arch) {
                            if let Some(feature) = feature {
                                let codegen_fn_attrs = self.tcx.codegen_fn_attrs(instance.def_id());
                                if self.tcx.sess.target_features.contains(&feature)
                                    || codegen_fn_attrs.target_features.contains(&feature)
                                {
                                    return true;
                                }
                            } else {
                                // Register class is unconditionally supported
                                return true;
                            }
                        }
                        false
                    };

                    let mut layout = None;
                    let ty = if let Some(ref place) = place {
                        layout = Some(&place.layout);
                        llvm_fixup_output_type(self.cx, reg.reg_class(), &place.layout)
                    } else if matches!(
                        reg.reg_class(),
                        InlineAsmRegClass::X86(
                            X86InlineAsmRegClass::mmx_reg | X86InlineAsmRegClass::x87_reg
                        )
                    ) {
                        // Special handling for x87/mmx registers: we always
                        // clobber the whole set if one register is marked as
                        // clobbered. This is due to the way LLVM handles the
                        // FP stack in inline assembly.
                        if !clobbered_x87 {
                            clobbered_x87 = true;
                            clobbers.push("~{st}".to_string());
                            for i in 1..=7 {
                                clobbers.push(format!("~{{st({})}}", i));
                            }
                        }
                        continue;
                    } else if !is_target_supported(reg.reg_class())
                        || reg.reg_class().is_clobber_only(asm_arch)
                    {
                        // We turn discarded outputs into clobber constraints
                        // if the target feature needed by the register class is
                        // disabled. This is necessary otherwise LLVM will try
                        // to actually allocate a register for the dummy output.
                        assert!(matches!(reg, InlineAsmRegOrRegClass::Reg(_)));
                        clobbers.push(format!("~{}", reg_to_llvm(reg, None)));
                        continue;
                    } else {
                        // If the output is discarded, we don't really care what
                        // type is used. We're just using this to tell LLVM to
                        // reserve the register.
                        dummy_output_type(self.cx, reg.reg_class())
                    };
                    output_types.push(ty);
                    op_idx.insert(idx, constraints.len());
                    let prefix = if late { "=" } else { "=&" };
                    constraints.push(format!("{}{}", prefix, reg_to_llvm(reg, layout)));
                }
                InlineAsmOperandRef::InOut { reg, late, in_value, out_place } => {
                    let layout = if let Some(ref out_place) = out_place {
                        &out_place.layout
                    } else {
                        // LLVM required tied operands to have the same type,
                        // so we just use the type of the input.
                        &in_value.layout
                    };
                    let ty = llvm_fixup_output_type(self.cx, reg.reg_class(), layout);
                    output_types.push(ty);
                    op_idx.insert(idx, constraints.len());
                    let prefix = if late { "=" } else { "=&" };
                    constraints.push(format!("{}{}", prefix, reg_to_llvm(reg, Some(layout))));
                }
                _ => {}
            }
        }

        // Collect input operands
        let mut inputs = vec![];
        for (idx, op) in operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::In { reg, value } => {
                    let llval =
                        llvm_fixup_input(self, value.immediate(), reg.reg_class(), &value.layout);
                    inputs.push(llval);
                    op_idx.insert(idx, constraints.len());
                    constraints.push(reg_to_llvm(reg, Some(&value.layout)));
                }
                InlineAsmOperandRef::InOut { reg, late: _, in_value, out_place: _ } => {
                    let value = llvm_fixup_input(
                        self,
                        in_value.immediate(),
                        reg.reg_class(),
                        &in_value.layout,
                    );
                    inputs.push(value);
                    constraints.push(format!("{}", op_idx[&idx]));
                }
                InlineAsmOperandRef::SymFn { instance } => {
                    inputs.push(self.cx.get_fn(instance));
                    op_idx.insert(idx, constraints.len());
                    constraints.push("s".to_string());
                }
                InlineAsmOperandRef::SymStatic { def_id } => {
                    inputs.push(self.cx.get_static(def_id));
                    op_idx.insert(idx, constraints.len());
                    constraints.push("s".to_string());
                }
                _ => {}
            }
        }

        // Build the template string
        let mut template_str = String::new();
        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => {
                    if s.contains('$') {
                        for c in s.chars() {
                            if c == '$' {
                                template_str.push_str("$$");
                            } else {
                                template_str.push(c);
                            }
                        }
                    } else {
                        template_str.push_str(s)
                    }
                }
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier, span: _ } => {
                    match operands[operand_idx] {
                        InlineAsmOperandRef::In { reg, .. }
                        | InlineAsmOperandRef::Out { reg, .. }
                        | InlineAsmOperandRef::InOut { reg, .. } => {
                            let modifier = modifier_to_llvm(asm_arch, reg.reg_class(), modifier);
                            if let Some(modifier) = modifier {
                                template_str.push_str(&format!(
                                    "${{{}:{}}}",
                                    op_idx[&operand_idx], modifier
                                ));
                            } else {
                                template_str.push_str(&format!("${{{}}}", op_idx[&operand_idx]));
                            }
                        }
                        InlineAsmOperandRef::Const { ref string } => {
                            // Const operands get injected directly into the template
                            template_str.push_str(string);
                        }
                        InlineAsmOperandRef::SymFn { .. }
                        | InlineAsmOperandRef::SymStatic { .. } => {
                            // Only emit the raw symbol name
                            template_str.push_str(&format!("${{{}:c}}", op_idx[&operand_idx]));
                        }
                    }
                }
            }
        }

        constraints.append(&mut clobbers);
        if !options.contains(InlineAsmOptions::PRESERVES_FLAGS) {
            match asm_arch {
                InlineAsmArch::AArch64 | InlineAsmArch::Arm => {
                    constraints.push("~{cc}".to_string());
                }
                InlineAsmArch::X86 | InlineAsmArch::X86_64 => {
                    constraints.extend_from_slice(&[
                        "~{dirflag}".to_string(),
                        "~{fpsr}".to_string(),
                        "~{flags}".to_string(),
                    ]);
                }
                InlineAsmArch::RiscV32 | InlineAsmArch::RiscV64 => {
                    constraints.extend_from_slice(&[
                        "~{vtype}".to_string(),
                        "~{vl}".to_string(),
                        "~{vxsat}".to_string(),
                        "~{vxrm}".to_string(),
                    ]);
                }
                InlineAsmArch::Avr => {
                    constraints.push("~{sreg}".to_string());
                }
                InlineAsmArch::Nvptx64 => {}
                InlineAsmArch::PowerPC | InlineAsmArch::PowerPC64 => {}
                InlineAsmArch::Hexagon => {}
                InlineAsmArch::Mips | InlineAsmArch::Mips64 => {}
                InlineAsmArch::S390x => {}
                InlineAsmArch::SpirV => {}
                InlineAsmArch::Wasm32 | InlineAsmArch::Wasm64 => {}
                InlineAsmArch::Bpf => {}
                InlineAsmArch::Msp430 => {
                    constraints.push("~{sr}".to_string());
                }
            }
        }
        if !options.contains(InlineAsmOptions::NOMEM) {
            // This is actually ignored by LLVM, but it's probably best to keep
            // it just in case. LLVM instead uses the ReadOnly/ReadNone
            // attributes on the call instruction to optimize.
            constraints.push("~{memory}".to_string());
        }
        let volatile = !options.contains(InlineAsmOptions::PURE);
        let alignstack = !options.contains(InlineAsmOptions::NOSTACK);
        let output_type = match &output_types[..] {
            [] => self.type_void(),
            [ty] => ty,
            tys => self.type_struct(tys, false),
        };
        let dialect = match asm_arch {
            InlineAsmArch::X86 | InlineAsmArch::X86_64
                if !options.contains(InlineAsmOptions::ATT_SYNTAX) =>
            {
                llvm::AsmDialect::Intel
            }
            _ => llvm::AsmDialect::Att,
        };
        let result = inline_asm_call(
            self,
            &template_str,
            &constraints.join(","),
            &inputs,
            output_type,
            volatile,
            alignstack,
            dialect,
            line_spans,
            options.contains(InlineAsmOptions::MAY_UNWIND),
            dest_catch_funclet,
        )
        .unwrap_or_else(|| span_bug!(line_spans[0], "LLVM asm constraint validation failed"));

        let mut attrs = SmallVec::<[_; 2]>::new();
        if options.contains(InlineAsmOptions::PURE) {
            if options.contains(InlineAsmOptions::NOMEM) {
                attrs.push(llvm::AttributeKind::ReadNone.create_attr(self.cx.llcx));
            } else if options.contains(InlineAsmOptions::READONLY) {
                attrs.push(llvm::AttributeKind::ReadOnly.create_attr(self.cx.llcx));
            }
            attrs.push(llvm::AttributeKind::WillReturn.create_attr(self.cx.llcx));
        } else if options.contains(InlineAsmOptions::NOMEM) {
            attrs.push(llvm::AttributeKind::InaccessibleMemOnly.create_attr(self.cx.llcx));
        } else {
            // LLVM doesn't have an attribute to represent ReadOnly + SideEffect
        }
        attributes::apply_to_callsite(result, llvm::AttributePlace::Function, &{ attrs });

        // Switch to the 'normal' basic block if we did an `invoke` instead of a `call`
        if let Some((dest, _, _)) = dest_catch_funclet {
            self.switch_to_block(dest);
        }

        // Write results to outputs
        for (idx, op) in operands.iter().enumerate() {
            if let InlineAsmOperandRef::Out { reg, place: Some(place), .. }
            | InlineAsmOperandRef::InOut { reg, out_place: Some(place), .. } = *op
            {
                let value = if output_types.len() == 1 {
                    result
                } else {
                    self.extract_value(result, op_idx[&idx] as u64)
                };
                let value = llvm_fixup_output(self, value, reg.reg_class(), &place.layout);
                OperandValue::Immediate(value).store(self, place);
            }
        }
    }
}

impl<'tcx> AsmMethods<'tcx> for CodegenCx<'_, 'tcx> {
    fn codegen_global_asm(
        &self,
        template: &[InlineAsmTemplatePiece],
        operands: &[GlobalAsmOperandRef<'tcx>],
        options: InlineAsmOptions,
        _line_spans: &[Span],
    ) {
        let asm_arch = self.tcx.sess.asm_arch.unwrap();

        // Default to Intel syntax on x86
        let intel_syntax = matches!(asm_arch, InlineAsmArch::X86 | InlineAsmArch::X86_64)
            && !options.contains(InlineAsmOptions::ATT_SYNTAX);

        // Build the template string
        let mut template_str = String::new();
        if intel_syntax {
            template_str.push_str(".intel_syntax\n");
        }
        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => template_str.push_str(s),
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier: _, span: _ } => {
                    match operands[operand_idx] {
                        GlobalAsmOperandRef::Const { ref string } => {
                            // Const operands get injected directly into the
                            // template. Note that we don't need to escape $
                            // here unlike normal inline assembly.
                            template_str.push_str(string);
                        }
                        GlobalAsmOperandRef::SymFn { instance } => {
                            let llval = self.get_fn(instance);
                            self.add_compiler_used_global(llval);
                            let symbol = llvm::build_string(|s| unsafe {
                                llvm::LLVMRustGetMangledName(llval, s);
                            })
                            .expect("symbol is not valid UTF-8");
                            template_str.push_str(&symbol);
                        }
                        GlobalAsmOperandRef::SymStatic { def_id } => {
                            let llval = self
                                .renamed_statics
                                .borrow()
                                .get(&def_id)
                                .copied()
                                .unwrap_or_else(|| self.get_static(def_id));
                            self.add_compiler_used_global(llval);
                            let symbol = llvm::build_string(|s| unsafe {
                                llvm::LLVMRustGetMangledName(llval, s);
                            })
                            .expect("symbol is not valid UTF-8");
                            template_str.push_str(&symbol);
                        }
                    }
                }
            }
        }
        if intel_syntax {
            template_str.push_str("\n.att_syntax\n");
        }

        unsafe {
            llvm::LLVMRustAppendModuleInlineAsm(
                self.llmod,
                template_str.as_ptr().cast(),
                template_str.len(),
            );
        }
    }
}

pub(crate) fn inline_asm_call<'ll>(
    bx: &mut Builder<'_, 'll, '_>,
    asm: &str,
    cons: &str,
    inputs: &[&'ll Value],
    output: &'ll llvm::Type,
    volatile: bool,
    alignstack: bool,
    dia: llvm::AsmDialect,
    line_spans: &[Span],
    unwind: bool,
    dest_catch_funclet: Option<(
        &'ll llvm::BasicBlock,
        &'ll llvm::BasicBlock,
        Option<&Funclet<'ll>>,
    )>,
) -> Option<&'ll Value> {
    let volatile = if volatile { llvm::True } else { llvm::False };
    let alignstack = if alignstack { llvm::True } else { llvm::False };
    let can_throw = if unwind { llvm::True } else { llvm::False };

    let argtys = inputs
        .iter()
        .map(|v| {
            debug!("Asm Input Type: {:?}", *v);
            bx.cx.val_ty(*v)
        })
        .collect::<Vec<_>>();

    debug!("Asm Output Type: {:?}", output);
    let fty = bx.cx.type_func(&argtys, output);
    unsafe {
        // Ask LLVM to verify that the constraints are well-formed.
        let constraints_ok = llvm::LLVMRustInlineAsmVerify(fty, cons.as_ptr().cast(), cons.len());
        debug!("constraint verification result: {:?}", constraints_ok);
        if constraints_ok {
            let v = llvm::LLVMRustInlineAsm(
                fty,
                asm.as_ptr().cast(),
                asm.len(),
                cons.as_ptr().cast(),
                cons.len(),
                volatile,
                alignstack,
                dia,
                can_throw,
            );

            let call = if let Some((dest, catch, funclet)) = dest_catch_funclet {
                bx.invoke(fty, v, inputs, dest, catch, funclet)
            } else {
                bx.call(fty, v, inputs, None)
            };

            // Store mark in a metadata node so we can map LLVM errors
            // back to source locations.  See #17552.
            let key = "srcloc";
            let kind = llvm::LLVMGetMDKindIDInContext(
                bx.llcx,
                key.as_ptr() as *const c_char,
                key.len() as c_uint,
            );

            // srcloc contains one integer for each line of assembly code.
            // Unfortunately this isn't enough to encode a full span so instead
            // we just encode the start position of each line.
            // FIXME: Figure out a way to pass the entire line spans.
            let mut srcloc = vec![];
            if dia == llvm::AsmDialect::Intel && line_spans.len() > 1 {
                // LLVM inserts an extra line to add the ".intel_syntax", so add
                // a dummy srcloc entry for it.
                //
                // Don't do this if we only have 1 line span since that may be
                // due to the asm template string coming from a macro. LLVM will
                // default to the first srcloc for lines that don't have an
                // associated srcloc.
                srcloc.push(bx.const_i32(0));
            }
            srcloc.extend(line_spans.iter().map(|span| bx.const_i32(span.lo().to_u32() as i32)));
            let md = llvm::LLVMMDNodeInContext(bx.llcx, srcloc.as_ptr(), srcloc.len() as u32);
            llvm::LLVMSetMetadata(call, kind, md);

            Some(call)
        } else {
            // LLVM has detected an issue with our constraints, bail out
            None
        }
    }
}

/// If the register is an xmm/ymm/zmm register then return its index.
fn xmm_reg_index(reg: InlineAsmReg) -> Option<u32> {
    match reg {
        InlineAsmReg::X86(reg)
            if reg as u32 >= X86InlineAsmReg::xmm0 as u32
                && reg as u32 <= X86InlineAsmReg::xmm15 as u32 =>
        {
            Some(reg as u32 - X86InlineAsmReg::xmm0 as u32)
        }
        InlineAsmReg::X86(reg)
            if reg as u32 >= X86InlineAsmReg::ymm0 as u32
                && reg as u32 <= X86InlineAsmReg::ymm15 as u32 =>
        {
            Some(reg as u32 - X86InlineAsmReg::ymm0 as u32)
        }
        InlineAsmReg::X86(reg)
            if reg as u32 >= X86InlineAsmReg::zmm0 as u32
                && reg as u32 <= X86InlineAsmReg::zmm31 as u32 =>
        {
            Some(reg as u32 - X86InlineAsmReg::zmm0 as u32)
        }
        _ => None,
    }
}

/// If the register is an AArch64 vector register then return its index.
fn a64_vreg_index(reg: InlineAsmReg) -> Option<u32> {
    match reg {
        InlineAsmReg::AArch64(reg)
            if reg as u32 >= AArch64InlineAsmReg::v0 as u32
                && reg as u32 <= AArch64InlineAsmReg::v31 as u32 =>
        {
            Some(reg as u32 - AArch64InlineAsmReg::v0 as u32)
        }
        _ => None,
    }
}

/// Converts a register class to an LLVM constraint code.
fn reg_to_llvm(reg: InlineAsmRegOrRegClass, layout: Option<&TyAndLayout<'_>>) -> String {
    match reg {
        // For vector registers LLVM wants the register name to match the type size.
        InlineAsmRegOrRegClass::Reg(reg) => {
            if let Some(idx) = xmm_reg_index(reg) {
                let class = if let Some(layout) = layout {
                    match layout.size.bytes() {
                        64 => 'z',
                        32 => 'y',
                        _ => 'x',
                    }
                } else {
                    // We use f32 as the type for discarded outputs
                    'x'
                };
                format!("{{{}mm{}}}", class, idx)
            } else if let Some(idx) = a64_vreg_index(reg) {
                let class = if let Some(layout) = layout {
                    match layout.size.bytes() {
                        16 => 'q',
                        8 => 'd',
                        4 => 's',
                        2 => 'h',
                        1 => 'd', // We fixup i8 to i8x8
                        _ => unreachable!(),
                    }
                } else {
                    // We use i64x2 as the type for discarded outputs
                    'q'
                };
                format!("{{{}{}}}", class, idx)
            } else if reg == InlineAsmReg::AArch64(AArch64InlineAsmReg::x30) {
                // LLVM doesn't recognize x30
                "{lr}".to_string()
            } else if reg == InlineAsmReg::Arm(ArmInlineAsmReg::r14) {
                // LLVM doesn't recognize r14
                "{lr}".to_string()
            } else {
                format!("{{{}}}", reg.name())
            }
        }
        InlineAsmRegOrRegClass::RegClass(reg) => match reg {
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg) => "w",
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16) => "x",
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::preg) => {
                unreachable!("clobber-only")
            }
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low16)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low8) => "t",
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg_low16)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low8)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low4) => "x",
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg) => "w",
            InlineAsmRegClass::Hexagon(HexagonInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Mips(MipsInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) => "h",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) => "r",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) => "l",
            InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::reg_nonzero) => "b",
            InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::cr)
            | InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::xer) => {
                unreachable!("clobber-only")
            }
            InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::vreg) => {
                unreachable!("clobber-only")
            }
            InlineAsmRegClass::X86(X86InlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd) => "Q",
            InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_byte) => "q",
            InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg)
            | InlineAsmRegClass::X86(X86InlineAsmRegClass::ymm_reg) => "x",
            InlineAsmRegClass::X86(X86InlineAsmRegClass::zmm_reg) => "v",
            InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg) => "^Yk",
            InlineAsmRegClass::X86(
                X86InlineAsmRegClass::x87_reg
                | X86InlineAsmRegClass::mmx_reg
                | X86InlineAsmRegClass::kreg0
                | X86InlineAsmRegClass::tmm_reg,
            ) => unreachable!("clobber-only"),
            InlineAsmRegClass::Wasm(WasmInlineAsmRegClass::local) => "r",
            InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::wreg) => "w",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_upper) => "d",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_pair) => "r",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_iw) => "w",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_ptr) => "e",
            InlineAsmRegClass::S390x(S390xInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::S390x(S390xInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::Msp430(Msp430InlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg) => {
                bug!("LLVM backend does not support SPIR-V")
            }
            InlineAsmRegClass::Err => unreachable!(),
        }
        .to_string(),
    }
}

/// Converts a modifier into LLVM's equivalent modifier.
fn modifier_to_llvm(
    arch: InlineAsmArch,
    reg: InlineAsmRegClass,
    modifier: Option<char>,
) -> Option<char> {
    match reg {
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::reg) => modifier,
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg)
        | InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16) => {
            if modifier == Some('v') { None } else { modifier }
        }
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::preg) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::reg) => None,
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg_low16) => None,
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low16)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low8) => Some('P'),
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low8)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low4) => {
            if modifier.is_none() {
                Some('q')
            } else {
                modifier
            }
        }
        InlineAsmRegClass::Hexagon(_) => None,
        InlineAsmRegClass::Mips(_) => None,
        InlineAsmRegClass::Nvptx(_) => None,
        InlineAsmRegClass::PowerPC(_) => None,
        InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::reg)
        | InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::freg) => None,
        InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::vreg) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::X86(X86InlineAsmRegClass::reg)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd) => match modifier {
            None if arch == InlineAsmArch::X86_64 => Some('q'),
            None => Some('k'),
            Some('l') => Some('b'),
            Some('h') => Some('h'),
            Some('x') => Some('w'),
            Some('e') => Some('k'),
            Some('r') => Some('q'),
            _ => unreachable!(),
        },
        InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_byte) => None,
        InlineAsmRegClass::X86(reg @ X86InlineAsmRegClass::xmm_reg)
        | InlineAsmRegClass::X86(reg @ X86InlineAsmRegClass::ymm_reg)
        | InlineAsmRegClass::X86(reg @ X86InlineAsmRegClass::zmm_reg) => match (reg, modifier) {
            (X86InlineAsmRegClass::xmm_reg, None) => Some('x'),
            (X86InlineAsmRegClass::ymm_reg, None) => Some('t'),
            (X86InlineAsmRegClass::zmm_reg, None) => Some('g'),
            (_, Some('x')) => Some('x'),
            (_, Some('y')) => Some('t'),
            (_, Some('z')) => Some('g'),
            _ => unreachable!(),
        },
        InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg) => None,
        InlineAsmRegClass::X86(
            X86InlineAsmRegClass::x87_reg
            | X86InlineAsmRegClass::mmx_reg
            | X86InlineAsmRegClass::kreg0
            | X86InlineAsmRegClass::tmm_reg,
        ) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::Wasm(WasmInlineAsmRegClass::local) => None,
        InlineAsmRegClass::Bpf(_) => None,
        InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_pair)
        | InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_iw)
        | InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_ptr) => match modifier {
            Some('h') => Some('B'),
            Some('l') => Some('A'),
            _ => None,
        },
        InlineAsmRegClass::Avr(_) => None,
        InlineAsmRegClass::S390x(_) => None,
        InlineAsmRegClass::Msp430(_) => None,
        InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg) => {
            bug!("LLVM backend does not support SPIR-V")
        }
        InlineAsmRegClass::Err => unreachable!(),
    }
}

/// Type to use for outputs that are discarded. It doesn't really matter what
/// the type is, as long as it is valid for the constraint code.
fn dummy_output_type<'ll>(cx: &CodegenCx<'ll, '_>, reg: InlineAsmRegClass) -> &'ll Type {
    match reg {
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg)
        | InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16) => {
            cx.type_vector(cx.type_i64(), 2)
        }
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::preg) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg_low16) => cx.type_f32(),
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low16)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low8) => cx.type_f64(),
        InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low8)
        | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low4) => {
            cx.type_vector(cx.type_i64(), 2)
        }
        InlineAsmRegClass::Hexagon(HexagonInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::Mips(MipsInlineAsmRegClass::freg) => cx.type_f32(),
        InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) => cx.type_i16(),
        InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) => cx.type_i32(),
        InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) => cx.type_i64(),
        InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::reg_nonzero) => cx.type_i32(),
        InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::freg) => cx.type_f64(),
        InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::cr)
        | InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::xer) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::freg) => cx.type_f32(),
        InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::vreg) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::X86(X86InlineAsmRegClass::reg)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd) => cx.type_i32(),
        InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_byte) => cx.type_i8(),
        InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::ymm_reg)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::zmm_reg) => cx.type_f32(),
        InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg) => cx.type_i16(),
        InlineAsmRegClass::X86(
            X86InlineAsmRegClass::x87_reg
            | X86InlineAsmRegClass::mmx_reg
            | X86InlineAsmRegClass::kreg0
            | X86InlineAsmRegClass::tmm_reg,
        ) => {
            unreachable!("clobber-only")
        }
        InlineAsmRegClass::Wasm(WasmInlineAsmRegClass::local) => cx.type_i32(),
        InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::reg) => cx.type_i64(),
        InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::wreg) => cx.type_i32(),
        InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg) => cx.type_i8(),
        InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_upper) => cx.type_i8(),
        InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_pair) => cx.type_i16(),
        InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_iw) => cx.type_i16(),
        InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_ptr) => cx.type_i16(),
        InlineAsmRegClass::S390x(S390xInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::S390x(S390xInlineAsmRegClass::freg) => cx.type_f64(),
        InlineAsmRegClass::Msp430(Msp430InlineAsmRegClass::reg) => cx.type_i16(),
        InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg) => {
            bug!("LLVM backend does not support SPIR-V")
        }
        InlineAsmRegClass::Err => unreachable!(),
    }
}

/// Helper function to get the LLVM type for a Scalar. Pointers are returned as
/// the equivalent integer type.
fn llvm_asm_scalar_type<'ll>(cx: &CodegenCx<'ll, '_>, scalar: Scalar) -> &'ll Type {
    match scalar.primitive() {
        Primitive::Int(Integer::I8, _) => cx.type_i8(),
        Primitive::Int(Integer::I16, _) => cx.type_i16(),
        Primitive::Int(Integer::I32, _) => cx.type_i32(),
        Primitive::Int(Integer::I64, _) => cx.type_i64(),
        Primitive::F32 => cx.type_f32(),
        Primitive::F64 => cx.type_f64(),
        Primitive::Pointer => cx.type_isize(),
        _ => unreachable!(),
    }
}

/// Fix up an input value to work around LLVM bugs.
fn llvm_fixup_input<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    mut value: &'ll Value,
    reg: InlineAsmRegClass,
    layout: &TyAndLayout<'tcx>,
) -> &'ll Value {
    match (reg, layout.abi) {
        (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg), Abi::Scalar(s)) => {
            if let Primitive::Int(Integer::I8, _) = s.primitive() {
                let vec_ty = bx.cx.type_vector(bx.cx.type_i8(), 8);
                bx.insert_element(bx.const_undef(vec_ty), value, bx.const_i32(0))
            } else {
                value
            }
        }
        (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16), Abi::Scalar(s)) => {
            let elem_ty = llvm_asm_scalar_type(bx.cx, s);
            let count = 16 / layout.size.bytes();
            let vec_ty = bx.cx.type_vector(elem_ty, count);
            if let Primitive::Pointer = s.primitive() {
                value = bx.ptrtoint(value, bx.cx.type_isize());
            }
            bx.insert_element(bx.const_undef(vec_ty), value, bx.const_i32(0))
        }
        (
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16),
            Abi::Vector { element, count },
        ) if layout.size.bytes() == 8 => {
            let elem_ty = llvm_asm_scalar_type(bx.cx, element);
            let vec_ty = bx.cx.type_vector(elem_ty, count);
            let indices: Vec<_> = (0..count * 2).map(|x| bx.const_i32(x as i32)).collect();
            bx.shuffle_vector(value, bx.const_undef(vec_ty), bx.const_vector(&indices))
        }
        (InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd), Abi::Scalar(s))
            if s.primitive() == Primitive::F64 =>
        {
            bx.bitcast(value, bx.cx.type_i64())
        }
        (
            InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg | X86InlineAsmRegClass::zmm_reg),
            Abi::Vector { .. },
        ) if layout.size.bytes() == 64 => bx.bitcast(value, bx.cx.type_vector(bx.cx.type_f64(), 8)),
        (
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg | ArmInlineAsmRegClass::sreg_low16),
            Abi::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I32, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_f32())
            } else {
                value
            }
        }
        (
            InlineAsmRegClass::Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16,
            ),
            Abi::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I64, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_f64())
            } else {
                value
            }
        }
        (InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg), Abi::Scalar(s)) => {
            match s.primitive() {
                // MIPS only supports register-length arithmetics.
                Primitive::Int(Integer::I8 | Integer::I16, _) => bx.zext(value, bx.cx.type_i32()),
                Primitive::F32 => bx.bitcast(value, bx.cx.type_i32()),
                Primitive::F64 => bx.bitcast(value, bx.cx.type_i64()),
                _ => value,
            }
        }
        _ => value,
    }
}

/// Fix up an output value to work around LLVM bugs.
fn llvm_fixup_output<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    mut value: &'ll Value,
    reg: InlineAsmRegClass,
    layout: &TyAndLayout<'tcx>,
) -> &'ll Value {
    match (reg, layout.abi) {
        (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg), Abi::Scalar(s)) => {
            if let Primitive::Int(Integer::I8, _) = s.primitive() {
                bx.extract_element(value, bx.const_i32(0))
            } else {
                value
            }
        }
        (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16), Abi::Scalar(s)) => {
            value = bx.extract_element(value, bx.const_i32(0));
            if let Primitive::Pointer = s.primitive() {
                value = bx.inttoptr(value, layout.llvm_type(bx.cx));
            }
            value
        }
        (
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16),
            Abi::Vector { element, count },
        ) if layout.size.bytes() == 8 => {
            let elem_ty = llvm_asm_scalar_type(bx.cx, element);
            let vec_ty = bx.cx.type_vector(elem_ty, count * 2);
            let indices: Vec<_> = (0..count).map(|x| bx.const_i32(x as i32)).collect();
            bx.shuffle_vector(value, bx.const_undef(vec_ty), bx.const_vector(&indices))
        }
        (InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd), Abi::Scalar(s))
            if s.primitive() == Primitive::F64 =>
        {
            bx.bitcast(value, bx.cx.type_f64())
        }
        (
            InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg | X86InlineAsmRegClass::zmm_reg),
            Abi::Vector { .. },
        ) if layout.size.bytes() == 64 => bx.bitcast(value, layout.llvm_type(bx.cx)),
        (
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg | ArmInlineAsmRegClass::sreg_low16),
            Abi::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I32, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_i32())
            } else {
                value
            }
        }
        (
            InlineAsmRegClass::Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16,
            ),
            Abi::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I64, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_i64())
            } else {
                value
            }
        }
        (InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg), Abi::Scalar(s)) => {
            match s.primitive() {
                // MIPS only supports register-length arithmetics.
                Primitive::Int(Integer::I8, _) => bx.trunc(value, bx.cx.type_i8()),
                Primitive::Int(Integer::I16, _) => bx.trunc(value, bx.cx.type_i16()),
                Primitive::F32 => bx.bitcast(value, bx.cx.type_f32()),
                Primitive::F64 => bx.bitcast(value, bx.cx.type_f64()),
                _ => value,
            }
        }
        _ => value,
    }
}

/// Output type to use for llvm_fixup_output.
fn llvm_fixup_output_type<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    reg: InlineAsmRegClass,
    layout: &TyAndLayout<'tcx>,
) -> &'ll Type {
    match (reg, layout.abi) {
        (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg), Abi::Scalar(s)) => {
            if let Primitive::Int(Integer::I8, _) = s.primitive() {
                cx.type_vector(cx.type_i8(), 8)
            } else {
                layout.llvm_type(cx)
            }
        }
        (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16), Abi::Scalar(s)) => {
            let elem_ty = llvm_asm_scalar_type(cx, s);
            let count = 16 / layout.size.bytes();
            cx.type_vector(elem_ty, count)
        }
        (
            InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16),
            Abi::Vector { element, count },
        ) if layout.size.bytes() == 8 => {
            let elem_ty = llvm_asm_scalar_type(cx, element);
            cx.type_vector(elem_ty, count * 2)
        }
        (InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd), Abi::Scalar(s))
            if s.primitive() == Primitive::F64 =>
        {
            cx.type_i64()
        }
        (
            InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg | X86InlineAsmRegClass::zmm_reg),
            Abi::Vector { .. },
        ) if layout.size.bytes() == 64 => cx.type_vector(cx.type_f64(), 8),
        (
            InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg | ArmInlineAsmRegClass::sreg_low16),
            Abi::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I32, _) = s.primitive() {
                cx.type_f32()
            } else {
                layout.llvm_type(cx)
            }
        }
        (
            InlineAsmRegClass::Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16,
            ),
            Abi::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I64, _) = s.primitive() {
                cx.type_f64()
            } else {
                layout.llvm_type(cx)
            }
        }
        (InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg), Abi::Scalar(s)) => {
            match s.primitive() {
                // MIPS only supports register-length arithmetics.
                Primitive::Int(Integer::I8 | Integer::I16, _) => cx.type_i32(),
                Primitive::F32 => cx.type_i32(),
                Primitive::F64 => cx.type_i64(),
                _ => layout.llvm_type(cx),
            }
        }
        _ => layout.llvm_type(cx),
    }
}
