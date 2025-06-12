use std::assert_matches::assert_matches;

use rustc_abi::{BackendRepr, Float, Integer, Primitive, Scalar};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::Instance;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::{bug, span_bug};
use rustc_span::{Pos, Span, Symbol, sym};
use rustc_target::asm::*;
use smallvec::SmallVec;
use tracing::debug;

use crate::builder::Builder;
use crate::common::Funclet;
use crate::context::CodegenCx;
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use crate::{attributes, llvm};

impl<'ll, 'tcx> AsmBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        line_spans: &[Span],
        instance: Instance<'_>,
        dest: Option<Self::BasicBlock>,
        catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
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
                        for &(_, feature) in reg_class.supported_types(asm_arch, true) {
                            if let Some(feature) = feature {
                                if self
                                    .tcx
                                    .asm_target_features(instance.def_id())
                                    .contains(&feature)
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
                        llvm_fixup_output_type(self.cx, reg.reg_class(), &place.layout, instance)
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
                        || reg.reg_class().is_clobber_only(asm_arch, true)
                    {
                        // We turn discarded outputs into clobber constraints
                        // if the target feature needed by the register class is
                        // disabled. This is necessary otherwise LLVM will try
                        // to actually allocate a register for the dummy output.
                        assert_matches!(reg, InlineAsmRegOrRegClass::Reg(_));
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
                    let ty = llvm_fixup_output_type(self.cx, reg.reg_class(), layout, instance);
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
                    let llval = llvm_fixup_input(
                        self,
                        value.immediate(),
                        reg.reg_class(),
                        &value.layout,
                        instance,
                    );
                    inputs.push(llval);
                    op_idx.insert(idx, constraints.len());
                    constraints.push(reg_to_llvm(reg, Some(&value.layout)));
                }
                InlineAsmOperandRef::InOut { reg, late, in_value, out_place: _ } => {
                    let value = llvm_fixup_input(
                        self,
                        in_value.immediate(),
                        reg.reg_class(),
                        &in_value.layout,
                        instance,
                    );
                    inputs.push(value);

                    // In the case of fixed registers, we have the choice of
                    // either using a tied operand or duplicating the constraint.
                    // We prefer the latter because it matches the behavior of
                    // Clang.
                    if late && matches!(reg, InlineAsmRegOrRegClass::Reg(_)) {
                        constraints.push(reg_to_llvm(reg, Some(&in_value.layout)));
                    } else {
                        constraints.push(format!("{}", op_idx[&idx]));
                    }
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
        let mut labels = vec![];
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
                        InlineAsmOperandRef::Label { label } => {
                            template_str.push_str(&format!("${{{}:l}}", constraints.len()));
                            constraints.push("!i".to_owned());
                            labels.push(label);
                        }
                    }
                }
            }
        }

        constraints.append(&mut clobbers);
        if !options.contains(InlineAsmOptions::PRESERVES_FLAGS) {
            match asm_arch {
                InlineAsmArch::AArch64 | InlineAsmArch::Arm64EC | InlineAsmArch::Arm => {
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
                InlineAsmArch::LoongArch32 | InlineAsmArch::LoongArch64 => {
                    constraints.extend_from_slice(&[
                        "~{$fcc0}".to_string(),
                        "~{$fcc1}".to_string(),
                        "~{$fcc2}".to_string(),
                        "~{$fcc3}".to_string(),
                        "~{$fcc4}".to_string(),
                        "~{$fcc5}".to_string(),
                        "~{$fcc6}".to_string(),
                        "~{$fcc7}".to_string(),
                    ]);
                }
                InlineAsmArch::Mips | InlineAsmArch::Mips64 => {}
                InlineAsmArch::S390x => {
                    constraints.push("~{cc}".to_string());
                }
                InlineAsmArch::Sparc | InlineAsmArch::Sparc64 => {
                    // In LLVM, ~{icc} represents icc and xcc in 64-bit code.
                    // https://github.com/llvm/llvm-project/blob/llvmorg-19.1.0/llvm/lib/Target/Sparc/SparcRegisterInfo.td#L64
                    constraints.push("~{icc}".to_string());
                    constraints.push("~{fcc0}".to_string());
                    constraints.push("~{fcc1}".to_string());
                    constraints.push("~{fcc2}".to_string());
                    constraints.push("~{fcc3}".to_string());
                }
                InlineAsmArch::SpirV => {}
                InlineAsmArch::Wasm32 | InlineAsmArch::Wasm64 => {}
                InlineAsmArch::Bpf => {}
                InlineAsmArch::Msp430 => {
                    constraints.push("~{sr}".to_string());
                }
                InlineAsmArch::M68k => {
                    constraints.push("~{ccr}".to_string());
                }
                InlineAsmArch::CSKY => {
                    constraints.push("~{psr}".to_string());
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
            &labels,
            volatile,
            alignstack,
            dialect,
            line_spans,
            options.contains(InlineAsmOptions::MAY_UNWIND),
            dest,
            catch_funclet,
        )
        .unwrap_or_else(|| span_bug!(line_spans[0], "LLVM asm constraint validation failed"));

        let mut attrs = SmallVec::<[_; 2]>::new();
        if options.contains(InlineAsmOptions::PURE) {
            if options.contains(InlineAsmOptions::NOMEM) {
                attrs.push(llvm::MemoryEffects::None.create_attr(self.cx.llcx));
            } else if options.contains(InlineAsmOptions::READONLY) {
                attrs.push(llvm::MemoryEffects::ReadOnly.create_attr(self.cx.llcx));
            }
            attrs.push(llvm::AttributeKind::WillReturn.create_attr(self.cx.llcx));
        } else if options.contains(InlineAsmOptions::NOMEM) {
            attrs.push(llvm::MemoryEffects::InaccessibleMemOnly.create_attr(self.cx.llcx));
        } else {
            // LLVM doesn't have an attribute to represent ReadOnly + SideEffect
        }
        attributes::apply_to_callsite(result, llvm::AttributePlace::Function, &{ attrs });

        // Write results to outputs. We need to do this for all possible control flow.
        //
        // Note that `dest` maybe populated with unreachable_block when asm goto with outputs
        // is used (because we need to codegen callbr which always needs a destination), so
        // here we use the NORETURN option to determine if `dest` should be used.
        for block in (if options.contains(InlineAsmOptions::NORETURN) { None } else { Some(dest) })
            .into_iter()
            .chain(labels.iter().copied().map(Some))
        {
            if let Some(block) = block {
                self.switch_to_block(block);
            }

            for (idx, op) in operands.iter().enumerate() {
                if let InlineAsmOperandRef::Out { reg, place: Some(place), .. }
                | InlineAsmOperandRef::InOut { reg, out_place: Some(place), .. } = *op
                {
                    let value = if output_types.len() == 1 {
                        result
                    } else {
                        self.extract_value(result, op_idx[&idx] as u64)
                    };
                    let value =
                        llvm_fixup_output(self, value, reg.reg_class(), &place.layout, instance);
                    OperandValue::Immediate(value).store(self, place);
                }
            }
        }
    }
}

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'_, 'tcx> {
    fn codegen_global_asm(
        &mut self,
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

        llvm::append_module_inline_asm(self.llmod, template_str.as_bytes());
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        let llval = self.get_fn(instance);
        llvm::build_string(|s| unsafe {
            llvm::LLVMRustGetMangledName(llval, s);
        })
        .expect("symbol is not valid UTF-8")
    }
}

pub(crate) fn inline_asm_call<'ll>(
    bx: &mut Builder<'_, 'll, '_>,
    asm: &str,
    cons: &str,
    inputs: &[&'ll Value],
    output: &'ll llvm::Type,
    labels: &[&'ll llvm::BasicBlock],
    volatile: bool,
    alignstack: bool,
    dia: llvm::AsmDialect,
    line_spans: &[Span],
    unwind: bool,
    dest: Option<&'ll llvm::BasicBlock>,
    catch_funclet: Option<(&'ll llvm::BasicBlock, Option<&Funclet<'ll>>)>,
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

    // Ask LLVM to verify that the constraints are well-formed.
    let constraints_ok = unsafe { llvm::LLVMRustInlineAsmVerify(fty, cons.as_ptr(), cons.len()) };
    debug!("constraint verification result: {:?}", constraints_ok);
    if !constraints_ok {
        // LLVM has detected an issue with our constraints, so bail out.
        return None;
    }

    let v = unsafe {
        llvm::LLVMGetInlineAsm(
            fty,
            asm.as_ptr(),
            asm.len(),
            cons.as_ptr(),
            cons.len(),
            volatile,
            alignstack,
            dia,
            can_throw,
        )
    };

    let call = if !labels.is_empty() {
        assert!(catch_funclet.is_none());
        bx.callbr(fty, None, None, v, inputs, dest.unwrap(), labels, None, None)
    } else if let Some((catch, funclet)) = catch_funclet {
        bx.invoke(fty, None, None, v, inputs, dest.unwrap(), catch, funclet, None)
    } else {
        bx.call(fty, None, None, v, inputs, None, None)
    };

    // Store mark in a metadata node so we can map LLVM errors
    // back to source locations. See #17552.
    let key = "srcloc";
    let kind = bx.get_md_kind_id(key);

    // `srcloc` contains one 64-bit integer for each line of assembly code,
    // where the lower 32 bits hold the lo byte position and the upper 32 bits
    // hold the hi byte position.
    let mut srcloc = vec![];
    if dia == llvm::AsmDialect::Intel && line_spans.len() > 1 {
        // LLVM inserts an extra line to add the ".intel_syntax", so add
        // a dummy srcloc entry for it.
        //
        // Don't do this if we only have 1 line span since that may be
        // due to the asm template string coming from a macro. LLVM will
        // default to the first srcloc for lines that don't have an
        // associated srcloc.
        srcloc.push(llvm::LLVMValueAsMetadata(bx.const_u64(0)));
    }
    srcloc.extend(line_spans.iter().map(|span| {
        llvm::LLVMValueAsMetadata(
            bx.const_u64(u64::from(span.lo().to_u32()) | (u64::from(span.hi().to_u32()) << 32)),
        )
    }));
    let md = unsafe { llvm::LLVMMDNodeInContext2(bx.llcx, srcloc.as_ptr(), srcloc.len()) };
    let md = bx.get_metadata_value(md);
    llvm::LLVMSetMetadata(call, kind, md);

    Some(call)
}

/// If the register is an xmm/ymm/zmm register then return its index.
fn xmm_reg_index(reg: InlineAsmReg) -> Option<u32> {
    use X86InlineAsmReg::*;
    match reg {
        InlineAsmReg::X86(reg) if reg as u32 >= xmm0 as u32 && reg as u32 <= xmm15 as u32 => {
            Some(reg as u32 - xmm0 as u32)
        }
        InlineAsmReg::X86(reg) if reg as u32 >= ymm0 as u32 && reg as u32 <= ymm15 as u32 => {
            Some(reg as u32 - ymm0 as u32)
        }
        InlineAsmReg::X86(reg) if reg as u32 >= zmm0 as u32 && reg as u32 <= zmm31 as u32 => {
            Some(reg as u32 - zmm0 as u32)
        }
        _ => None,
    }
}

/// If the register is an AArch64 integer register then return its index.
fn a64_reg_index(reg: InlineAsmReg) -> Option<u32> {
    match reg {
        InlineAsmReg::AArch64(r) => r.reg_index(),
        _ => None,
    }
}

/// If the register is an AArch64 vector register then return its index.
fn a64_vreg_index(reg: InlineAsmReg) -> Option<u32> {
    match reg {
        InlineAsmReg::AArch64(reg) => reg.vreg_index(),
        _ => None,
    }
}

/// Converts a register class to an LLVM constraint code.
fn reg_to_llvm(reg: InlineAsmRegOrRegClass, layout: Option<&TyAndLayout<'_>>) -> String {
    use InlineAsmRegClass::*;
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
            } else if let Some(idx) = a64_reg_index(reg) {
                let class = if let Some(layout) = layout {
                    match layout.size.bytes() {
                        8 => 'x',
                        _ => 'w',
                    }
                } else {
                    // We use i32 as the type for discarded outputs
                    'w'
                };
                if class == 'x' && reg == InlineAsmReg::AArch64(AArch64InlineAsmReg::x30) {
                    // LLVM doesn't recognize x30. use lr instead.
                    "{lr}".to_string()
                } else {
                    format!("{{{}{}}}", class, idx)
                }
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
            } else if reg == InlineAsmReg::Arm(ArmInlineAsmReg::r14) {
                // LLVM doesn't recognize r14
                "{lr}".to_string()
            } else {
                format!("{{{}}}", reg.name())
            }
        }
        // The constraints can be retrieved from
        // https://llvm.org/docs/LangRef.html#supported-constraint-code-list
        InlineAsmRegOrRegClass::RegClass(reg) => match reg {
            AArch64(AArch64InlineAsmRegClass::reg) => "r",
            AArch64(AArch64InlineAsmRegClass::vreg) => "w",
            AArch64(AArch64InlineAsmRegClass::vreg_low16) => "x",
            AArch64(AArch64InlineAsmRegClass::preg) => unreachable!("clobber-only"),
            Arm(ArmInlineAsmRegClass::reg) => "r",
            Arm(ArmInlineAsmRegClass::sreg)
            | Arm(ArmInlineAsmRegClass::dreg_low16)
            | Arm(ArmInlineAsmRegClass::qreg_low8) => "t",
            Arm(ArmInlineAsmRegClass::sreg_low16)
            | Arm(ArmInlineAsmRegClass::dreg_low8)
            | Arm(ArmInlineAsmRegClass::qreg_low4) => "x",
            Arm(ArmInlineAsmRegClass::dreg) | Arm(ArmInlineAsmRegClass::qreg) => "w",
            Hexagon(HexagonInlineAsmRegClass::reg) => "r",
            Hexagon(HexagonInlineAsmRegClass::preg) => unreachable!("clobber-only"),
            LoongArch(LoongArchInlineAsmRegClass::reg) => "r",
            LoongArch(LoongArchInlineAsmRegClass::freg) => "f",
            Mips(MipsInlineAsmRegClass::reg) => "r",
            Mips(MipsInlineAsmRegClass::freg) => "f",
            Nvptx(NvptxInlineAsmRegClass::reg16) => "h",
            Nvptx(NvptxInlineAsmRegClass::reg32) => "r",
            Nvptx(NvptxInlineAsmRegClass::reg64) => "l",
            PowerPC(PowerPCInlineAsmRegClass::reg) => "r",
            PowerPC(PowerPCInlineAsmRegClass::reg_nonzero) => "b",
            PowerPC(PowerPCInlineAsmRegClass::freg) => "f",
            PowerPC(PowerPCInlineAsmRegClass::vreg) => "v",
            PowerPC(PowerPCInlineAsmRegClass::cr) | PowerPC(PowerPCInlineAsmRegClass::xer) => {
                unreachable!("clobber-only")
            }
            RiscV(RiscVInlineAsmRegClass::reg) => "r",
            RiscV(RiscVInlineAsmRegClass::freg) => "f",
            RiscV(RiscVInlineAsmRegClass::vreg) => unreachable!("clobber-only"),
            X86(X86InlineAsmRegClass::reg) => "r",
            X86(X86InlineAsmRegClass::reg_abcd) => "Q",
            X86(X86InlineAsmRegClass::reg_byte) => "q",
            X86(X86InlineAsmRegClass::xmm_reg) | X86(X86InlineAsmRegClass::ymm_reg) => "x",
            X86(X86InlineAsmRegClass::zmm_reg) => "v",
            X86(X86InlineAsmRegClass::kreg) => "^Yk",
            X86(
                X86InlineAsmRegClass::x87_reg
                | X86InlineAsmRegClass::mmx_reg
                | X86InlineAsmRegClass::kreg0
                | X86InlineAsmRegClass::tmm_reg,
            ) => unreachable!("clobber-only"),
            Wasm(WasmInlineAsmRegClass::local) => "r",
            Bpf(BpfInlineAsmRegClass::reg) => "r",
            Bpf(BpfInlineAsmRegClass::wreg) => "w",
            Avr(AvrInlineAsmRegClass::reg) => "r",
            Avr(AvrInlineAsmRegClass::reg_upper) => "d",
            Avr(AvrInlineAsmRegClass::reg_pair) => "r",
            Avr(AvrInlineAsmRegClass::reg_iw) => "w",
            Avr(AvrInlineAsmRegClass::reg_ptr) => "e",
            S390x(S390xInlineAsmRegClass::reg) => "r",
            S390x(S390xInlineAsmRegClass::reg_addr) => "a",
            S390x(S390xInlineAsmRegClass::freg) => "f",
            S390x(S390xInlineAsmRegClass::vreg) => "v",
            S390x(S390xInlineAsmRegClass::areg) => {
                unreachable!("clobber-only")
            }
            Sparc(SparcInlineAsmRegClass::reg) => "r",
            Sparc(SparcInlineAsmRegClass::yreg) => unreachable!("clobber-only"),
            Msp430(Msp430InlineAsmRegClass::reg) => "r",
            M68k(M68kInlineAsmRegClass::reg) => "r",
            M68k(M68kInlineAsmRegClass::reg_addr) => "a",
            M68k(M68kInlineAsmRegClass::reg_data) => "d",
            CSKY(CSKYInlineAsmRegClass::reg) => "r",
            CSKY(CSKYInlineAsmRegClass::freg) => "f",
            SpirV(SpirVInlineAsmRegClass::reg) => bug!("LLVM backend does not support SPIR-V"),
            Err => unreachable!(),
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
    use InlineAsmRegClass::*;
    // The modifiers can be retrieved from
    // https://llvm.org/docs/LangRef.html#asm-template-argument-modifiers
    match reg {
        AArch64(AArch64InlineAsmRegClass::reg) => modifier,
        AArch64(AArch64InlineAsmRegClass::vreg) | AArch64(AArch64InlineAsmRegClass::vreg_low16) => {
            if modifier == Some('v') {
                None
            } else {
                modifier
            }
        }
        AArch64(AArch64InlineAsmRegClass::preg) => unreachable!("clobber-only"),
        Arm(ArmInlineAsmRegClass::reg) => None,
        Arm(ArmInlineAsmRegClass::sreg) | Arm(ArmInlineAsmRegClass::sreg_low16) => None,
        Arm(ArmInlineAsmRegClass::dreg)
        | Arm(ArmInlineAsmRegClass::dreg_low16)
        | Arm(ArmInlineAsmRegClass::dreg_low8) => Some('P'),
        Arm(ArmInlineAsmRegClass::qreg)
        | Arm(ArmInlineAsmRegClass::qreg_low8)
        | Arm(ArmInlineAsmRegClass::qreg_low4) => {
            if modifier.is_none() {
                Some('q')
            } else {
                modifier
            }
        }
        Hexagon(_) => None,
        LoongArch(_) => None,
        Mips(_) => None,
        Nvptx(_) => None,
        PowerPC(_) => None,
        RiscV(RiscVInlineAsmRegClass::reg) | RiscV(RiscVInlineAsmRegClass::freg) => None,
        RiscV(RiscVInlineAsmRegClass::vreg) => unreachable!("clobber-only"),
        X86(X86InlineAsmRegClass::reg) | X86(X86InlineAsmRegClass::reg_abcd) => match modifier {
            None if arch == InlineAsmArch::X86_64 => Some('q'),
            None => Some('k'),
            Some('l') => Some('b'),
            Some('h') => Some('h'),
            Some('x') => Some('w'),
            Some('e') => Some('k'),
            Some('r') => Some('q'),
            _ => unreachable!(),
        },
        X86(X86InlineAsmRegClass::reg_byte) => None,
        X86(reg @ X86InlineAsmRegClass::xmm_reg)
        | X86(reg @ X86InlineAsmRegClass::ymm_reg)
        | X86(reg @ X86InlineAsmRegClass::zmm_reg) => match (reg, modifier) {
            (X86InlineAsmRegClass::xmm_reg, None) => Some('x'),
            (X86InlineAsmRegClass::ymm_reg, None) => Some('t'),
            (X86InlineAsmRegClass::zmm_reg, None) => Some('g'),
            (_, Some('x')) => Some('x'),
            (_, Some('y')) => Some('t'),
            (_, Some('z')) => Some('g'),
            _ => unreachable!(),
        },
        X86(X86InlineAsmRegClass::kreg) => None,
        X86(
            X86InlineAsmRegClass::x87_reg
            | X86InlineAsmRegClass::mmx_reg
            | X86InlineAsmRegClass::kreg0
            | X86InlineAsmRegClass::tmm_reg,
        ) => unreachable!("clobber-only"),
        Wasm(WasmInlineAsmRegClass::local) => None,
        Bpf(_) => None,
        Avr(AvrInlineAsmRegClass::reg_pair)
        | Avr(AvrInlineAsmRegClass::reg_iw)
        | Avr(AvrInlineAsmRegClass::reg_ptr) => match modifier {
            Some('h') => Some('B'),
            Some('l') => Some('A'),
            _ => None,
        },
        Avr(_) => None,
        S390x(_) => None,
        Sparc(_) => None,
        Msp430(_) => None,
        SpirV(SpirVInlineAsmRegClass::reg) => bug!("LLVM backend does not support SPIR-V"),
        M68k(_) => None,
        CSKY(_) => None,
        Err => unreachable!(),
    }
}

/// Type to use for outputs that are discarded. It doesn't really matter what
/// the type is, as long as it is valid for the constraint code.
fn dummy_output_type<'ll>(cx: &CodegenCx<'ll, '_>, reg: InlineAsmRegClass) -> &'ll Type {
    use InlineAsmRegClass::*;
    match reg {
        AArch64(AArch64InlineAsmRegClass::reg) => cx.type_i32(),
        AArch64(AArch64InlineAsmRegClass::vreg) | AArch64(AArch64InlineAsmRegClass::vreg_low16) => {
            cx.type_vector(cx.type_i64(), 2)
        }
        AArch64(AArch64InlineAsmRegClass::preg) => unreachable!("clobber-only"),
        Arm(ArmInlineAsmRegClass::reg) => cx.type_i32(),
        Arm(ArmInlineAsmRegClass::sreg) | Arm(ArmInlineAsmRegClass::sreg_low16) => cx.type_f32(),
        Arm(ArmInlineAsmRegClass::dreg)
        | Arm(ArmInlineAsmRegClass::dreg_low16)
        | Arm(ArmInlineAsmRegClass::dreg_low8) => cx.type_f64(),
        Arm(ArmInlineAsmRegClass::qreg)
        | Arm(ArmInlineAsmRegClass::qreg_low8)
        | Arm(ArmInlineAsmRegClass::qreg_low4) => cx.type_vector(cx.type_i64(), 2),
        Hexagon(HexagonInlineAsmRegClass::reg) => cx.type_i32(),
        Hexagon(HexagonInlineAsmRegClass::preg) => unreachable!("clobber-only"),
        LoongArch(LoongArchInlineAsmRegClass::reg) => cx.type_i32(),
        LoongArch(LoongArchInlineAsmRegClass::freg) => cx.type_f32(),
        Mips(MipsInlineAsmRegClass::reg) => cx.type_i32(),
        Mips(MipsInlineAsmRegClass::freg) => cx.type_f32(),
        Nvptx(NvptxInlineAsmRegClass::reg16) => cx.type_i16(),
        Nvptx(NvptxInlineAsmRegClass::reg32) => cx.type_i32(),
        Nvptx(NvptxInlineAsmRegClass::reg64) => cx.type_i64(),
        PowerPC(PowerPCInlineAsmRegClass::reg) => cx.type_i32(),
        PowerPC(PowerPCInlineAsmRegClass::reg_nonzero) => cx.type_i32(),
        PowerPC(PowerPCInlineAsmRegClass::freg) => cx.type_f64(),
        PowerPC(PowerPCInlineAsmRegClass::vreg) => cx.type_vector(cx.type_i32(), 4),
        PowerPC(PowerPCInlineAsmRegClass::cr) | PowerPC(PowerPCInlineAsmRegClass::xer) => {
            unreachable!("clobber-only")
        }
        RiscV(RiscVInlineAsmRegClass::reg) => cx.type_i32(),
        RiscV(RiscVInlineAsmRegClass::freg) => cx.type_f32(),
        RiscV(RiscVInlineAsmRegClass::vreg) => unreachable!("clobber-only"),
        X86(X86InlineAsmRegClass::reg) | X86(X86InlineAsmRegClass::reg_abcd) => cx.type_i32(),
        X86(X86InlineAsmRegClass::reg_byte) => cx.type_i8(),
        X86(X86InlineAsmRegClass::xmm_reg)
        | X86(X86InlineAsmRegClass::ymm_reg)
        | X86(X86InlineAsmRegClass::zmm_reg) => cx.type_f32(),
        X86(X86InlineAsmRegClass::kreg) => cx.type_i16(),
        X86(
            X86InlineAsmRegClass::x87_reg
            | X86InlineAsmRegClass::mmx_reg
            | X86InlineAsmRegClass::kreg0
            | X86InlineAsmRegClass::tmm_reg,
        ) => unreachable!("clobber-only"),
        Wasm(WasmInlineAsmRegClass::local) => cx.type_i32(),
        Bpf(BpfInlineAsmRegClass::reg) => cx.type_i64(),
        Bpf(BpfInlineAsmRegClass::wreg) => cx.type_i32(),
        Avr(AvrInlineAsmRegClass::reg) => cx.type_i8(),
        Avr(AvrInlineAsmRegClass::reg_upper) => cx.type_i8(),
        Avr(AvrInlineAsmRegClass::reg_pair) => cx.type_i16(),
        Avr(AvrInlineAsmRegClass::reg_iw) => cx.type_i16(),
        Avr(AvrInlineAsmRegClass::reg_ptr) => cx.type_i16(),
        S390x(S390xInlineAsmRegClass::reg | S390xInlineAsmRegClass::reg_addr) => cx.type_i32(),
        S390x(S390xInlineAsmRegClass::freg) => cx.type_f64(),
        S390x(S390xInlineAsmRegClass::vreg) => cx.type_vector(cx.type_i64(), 2),
        S390x(S390xInlineAsmRegClass::areg) => {
            unreachable!("clobber-only")
        }
        Sparc(SparcInlineAsmRegClass::reg) => cx.type_i32(),
        Sparc(SparcInlineAsmRegClass::yreg) => unreachable!("clobber-only"),
        Msp430(Msp430InlineAsmRegClass::reg) => cx.type_i16(),
        M68k(M68kInlineAsmRegClass::reg) => cx.type_i32(),
        M68k(M68kInlineAsmRegClass::reg_addr) => cx.type_i32(),
        M68k(M68kInlineAsmRegClass::reg_data) => cx.type_i32(),
        CSKY(CSKYInlineAsmRegClass::reg) => cx.type_i32(),
        CSKY(CSKYInlineAsmRegClass::freg) => cx.type_f32(),
        SpirV(SpirVInlineAsmRegClass::reg) => bug!("LLVM backend does not support SPIR-V"),
        Err => unreachable!(),
    }
}

/// Helper function to get the LLVM type for a Scalar. Pointers are returned as
/// the equivalent integer type.
fn llvm_asm_scalar_type<'ll>(cx: &CodegenCx<'ll, '_>, scalar: Scalar) -> &'ll Type {
    let dl = &cx.tcx.data_layout;
    match scalar.primitive() {
        Primitive::Int(Integer::I8, _) => cx.type_i8(),
        Primitive::Int(Integer::I16, _) => cx.type_i16(),
        Primitive::Int(Integer::I32, _) => cx.type_i32(),
        Primitive::Int(Integer::I64, _) => cx.type_i64(),
        Primitive::Float(Float::F16) => cx.type_f16(),
        Primitive::Float(Float::F32) => cx.type_f32(),
        Primitive::Float(Float::F64) => cx.type_f64(),
        Primitive::Float(Float::F128) => cx.type_f128(),
        // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
        Primitive::Pointer(_) => cx.type_from_integer(dl.ptr_sized_integer()),
        _ => unreachable!(),
    }
}

fn any_target_feature_enabled(
    cx: &CodegenCx<'_, '_>,
    instance: Instance<'_>,
    features: &[Symbol],
) -> bool {
    let enabled = cx.tcx.asm_target_features(instance.def_id());
    features.iter().any(|feat| enabled.contains(feat))
}

/// Fix up an input value to work around LLVM bugs.
fn llvm_fixup_input<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    mut value: &'ll Value,
    reg: InlineAsmRegClass,
    layout: &TyAndLayout<'tcx>,
    instance: Instance<'_>,
) -> &'ll Value {
    use InlineAsmRegClass::*;
    let dl = &bx.tcx.data_layout;
    match (reg, layout.backend_repr) {
        (AArch64(AArch64InlineAsmRegClass::vreg), BackendRepr::Scalar(s)) => {
            if let Primitive::Int(Integer::I8, _) = s.primitive() {
                let vec_ty = bx.cx.type_vector(bx.cx.type_i8(), 8);
                bx.insert_element(bx.const_undef(vec_ty), value, bx.const_i32(0))
            } else {
                value
            }
        }
        (AArch64(AArch64InlineAsmRegClass::vreg_low16), BackendRepr::Scalar(s))
            if s.primitive() != Primitive::Float(Float::F128) =>
        {
            let elem_ty = llvm_asm_scalar_type(bx.cx, s);
            let count = 16 / layout.size.bytes();
            let vec_ty = bx.cx.type_vector(elem_ty, count);
            // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
            if let Primitive::Pointer(_) = s.primitive() {
                let t = bx.type_from_integer(dl.ptr_sized_integer());
                value = bx.ptrtoint(value, t);
            }
            bx.insert_element(bx.const_undef(vec_ty), value, bx.const_i32(0))
        }
        (
            AArch64(AArch64InlineAsmRegClass::vreg_low16),
            BackendRepr::SimdVector { element, count },
        ) if layout.size.bytes() == 8 => {
            let elem_ty = llvm_asm_scalar_type(bx.cx, element);
            let vec_ty = bx.cx.type_vector(elem_ty, count);
            let indices: Vec<_> = (0..count * 2).map(|x| bx.const_i32(x as i32)).collect();
            bx.shuffle_vector(value, bx.const_undef(vec_ty), bx.const_vector(&indices))
        }
        (X86(X86InlineAsmRegClass::reg_abcd), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F64) =>
        {
            bx.bitcast(value, bx.cx.type_i64())
        }
        (
            X86(X86InlineAsmRegClass::xmm_reg | X86InlineAsmRegClass::zmm_reg),
            BackendRepr::SimdVector { .. },
        ) if layout.size.bytes() == 64 => bx.bitcast(value, bx.cx.type_vector(bx.cx.type_f64(), 8)),
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::Scalar(s),
        ) if bx.sess().asm_arch == Some(InlineAsmArch::X86)
            && s.primitive() == Primitive::Float(Float::F128) =>
        {
            bx.bitcast(value, bx.type_vector(bx.type_i32(), 4))
        }
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::Scalar(s),
        ) if s.primitive() == Primitive::Float(Float::F16) => {
            let value = bx.insert_element(
                bx.const_undef(bx.type_vector(bx.type_f16(), 8)),
                value,
                bx.const_usize(0),
            );
            bx.bitcast(value, bx.type_vector(bx.type_i16(), 8))
        }
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::SimdVector { element, count: count @ (8 | 16) },
        ) if element.primitive() == Primitive::Float(Float::F16) => {
            bx.bitcast(value, bx.type_vector(bx.type_i16(), count))
        }
        (
            Arm(ArmInlineAsmRegClass::sreg | ArmInlineAsmRegClass::sreg_low16),
            BackendRepr::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I32, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_f32())
            } else {
                value
            }
        }
        (
            Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16,
            ),
            BackendRepr::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I64, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_f64())
            } else {
                value
            }
        }
        (
            Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16
                | ArmInlineAsmRegClass::qreg
                | ArmInlineAsmRegClass::qreg_low4
                | ArmInlineAsmRegClass::qreg_low8,
            ),
            BackendRepr::SimdVector { element, count: count @ (4 | 8) },
        ) if element.primitive() == Primitive::Float(Float::F16) => {
            bx.bitcast(value, bx.type_vector(bx.type_i16(), count))
        }
        (LoongArch(LoongArchInlineAsmRegClass::freg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F16) =>
        {
            // Smaller floats are always "NaN-boxed" inside larger floats on LoongArch.
            let value = bx.bitcast(value, bx.type_i16());
            let value = bx.zext(value, bx.type_i32());
            let value = bx.or(value, bx.const_u32(0xFFFF_0000));
            bx.bitcast(value, bx.type_f32())
        }
        (Mips(MipsInlineAsmRegClass::reg), BackendRepr::Scalar(s)) => {
            match s.primitive() {
                // MIPS only supports register-length arithmetics.
                Primitive::Int(Integer::I8 | Integer::I16, _) => bx.zext(value, bx.cx.type_i32()),
                Primitive::Float(Float::F32) => bx.bitcast(value, bx.cx.type_i32()),
                Primitive::Float(Float::F64) => bx.bitcast(value, bx.cx.type_i64()),
                _ => value,
            }
        }
        (RiscV(RiscVInlineAsmRegClass::freg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F16)
                && !any_target_feature_enabled(bx, instance, &[sym::zfhmin, sym::zfh]) =>
        {
            // Smaller floats are always "NaN-boxed" inside larger floats on RISC-V.
            let value = bx.bitcast(value, bx.type_i16());
            let value = bx.zext(value, bx.type_i32());
            let value = bx.or(value, bx.const_u32(0xFFFF_0000));
            bx.bitcast(value, bx.type_f32())
        }
        (PowerPC(PowerPCInlineAsmRegClass::vreg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F32) =>
        {
            let value = bx.insert_element(
                bx.const_undef(bx.type_vector(bx.type_f32(), 4)),
                value,
                bx.const_usize(0),
            );
            bx.bitcast(value, bx.type_vector(bx.type_f32(), 4))
        }
        (PowerPC(PowerPCInlineAsmRegClass::vreg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F64) =>
        {
            let value = bx.insert_element(
                bx.const_undef(bx.type_vector(bx.type_f64(), 2)),
                value,
                bx.const_usize(0),
            );
            bx.bitcast(value, bx.type_vector(bx.type_f64(), 2))
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
    instance: Instance<'_>,
) -> &'ll Value {
    use InlineAsmRegClass::*;
    match (reg, layout.backend_repr) {
        (AArch64(AArch64InlineAsmRegClass::vreg), BackendRepr::Scalar(s)) => {
            if let Primitive::Int(Integer::I8, _) = s.primitive() {
                bx.extract_element(value, bx.const_i32(0))
            } else {
                value
            }
        }
        (AArch64(AArch64InlineAsmRegClass::vreg_low16), BackendRepr::Scalar(s))
            if s.primitive() != Primitive::Float(Float::F128) =>
        {
            value = bx.extract_element(value, bx.const_i32(0));
            if let Primitive::Pointer(_) = s.primitive() {
                value = bx.inttoptr(value, layout.llvm_type(bx.cx));
            }
            value
        }
        (
            AArch64(AArch64InlineAsmRegClass::vreg_low16),
            BackendRepr::SimdVector { element, count },
        ) if layout.size.bytes() == 8 => {
            let elem_ty = llvm_asm_scalar_type(bx.cx, element);
            let vec_ty = bx.cx.type_vector(elem_ty, count * 2);
            let indices: Vec<_> = (0..count).map(|x| bx.const_i32(x as i32)).collect();
            bx.shuffle_vector(value, bx.const_undef(vec_ty), bx.const_vector(&indices))
        }
        (X86(X86InlineAsmRegClass::reg_abcd), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F64) =>
        {
            bx.bitcast(value, bx.cx.type_f64())
        }
        (
            X86(X86InlineAsmRegClass::xmm_reg | X86InlineAsmRegClass::zmm_reg),
            BackendRepr::SimdVector { .. },
        ) if layout.size.bytes() == 64 => bx.bitcast(value, layout.llvm_type(bx.cx)),
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::Scalar(s),
        ) if bx.sess().asm_arch == Some(InlineAsmArch::X86)
            && s.primitive() == Primitive::Float(Float::F128) =>
        {
            bx.bitcast(value, bx.type_f128())
        }
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::Scalar(s),
        ) if s.primitive() == Primitive::Float(Float::F16) => {
            let value = bx.bitcast(value, bx.type_vector(bx.type_f16(), 8));
            bx.extract_element(value, bx.const_usize(0))
        }
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::SimdVector { element, count: count @ (8 | 16) },
        ) if element.primitive() == Primitive::Float(Float::F16) => {
            bx.bitcast(value, bx.type_vector(bx.type_f16(), count))
        }
        (
            Arm(ArmInlineAsmRegClass::sreg | ArmInlineAsmRegClass::sreg_low16),
            BackendRepr::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I32, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_i32())
            } else {
                value
            }
        }
        (
            Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16,
            ),
            BackendRepr::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I64, _) = s.primitive() {
                bx.bitcast(value, bx.cx.type_i64())
            } else {
                value
            }
        }
        (
            Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16
                | ArmInlineAsmRegClass::qreg
                | ArmInlineAsmRegClass::qreg_low4
                | ArmInlineAsmRegClass::qreg_low8,
            ),
            BackendRepr::SimdVector { element, count: count @ (4 | 8) },
        ) if element.primitive() == Primitive::Float(Float::F16) => {
            bx.bitcast(value, bx.type_vector(bx.type_f16(), count))
        }
        (LoongArch(LoongArchInlineAsmRegClass::freg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F16) =>
        {
            let value = bx.bitcast(value, bx.type_i32());
            let value = bx.trunc(value, bx.type_i16());
            bx.bitcast(value, bx.type_f16())
        }
        (Mips(MipsInlineAsmRegClass::reg), BackendRepr::Scalar(s)) => {
            match s.primitive() {
                // MIPS only supports register-length arithmetics.
                Primitive::Int(Integer::I8, _) => bx.trunc(value, bx.cx.type_i8()),
                Primitive::Int(Integer::I16, _) => bx.trunc(value, bx.cx.type_i16()),
                Primitive::Float(Float::F32) => bx.bitcast(value, bx.cx.type_f32()),
                Primitive::Float(Float::F64) => bx.bitcast(value, bx.cx.type_f64()),
                _ => value,
            }
        }
        (RiscV(RiscVInlineAsmRegClass::freg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F16)
                && !any_target_feature_enabled(bx, instance, &[sym::zfhmin, sym::zfh]) =>
        {
            let value = bx.bitcast(value, bx.type_i32());
            let value = bx.trunc(value, bx.type_i16());
            bx.bitcast(value, bx.type_f16())
        }
        (PowerPC(PowerPCInlineAsmRegClass::vreg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F32) =>
        {
            let value = bx.bitcast(value, bx.type_vector(bx.type_f32(), 4));
            bx.extract_element(value, bx.const_usize(0))
        }
        (PowerPC(PowerPCInlineAsmRegClass::vreg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F64) =>
        {
            let value = bx.bitcast(value, bx.type_vector(bx.type_f64(), 2));
            bx.extract_element(value, bx.const_usize(0))
        }
        _ => value,
    }
}

/// Output type to use for llvm_fixup_output.
fn llvm_fixup_output_type<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    reg: InlineAsmRegClass,
    layout: &TyAndLayout<'tcx>,
    instance: Instance<'_>,
) -> &'ll Type {
    use InlineAsmRegClass::*;
    match (reg, layout.backend_repr) {
        (AArch64(AArch64InlineAsmRegClass::vreg), BackendRepr::Scalar(s)) => {
            if let Primitive::Int(Integer::I8, _) = s.primitive() {
                cx.type_vector(cx.type_i8(), 8)
            } else {
                layout.llvm_type(cx)
            }
        }
        (AArch64(AArch64InlineAsmRegClass::vreg_low16), BackendRepr::Scalar(s))
            if s.primitive() != Primitive::Float(Float::F128) =>
        {
            let elem_ty = llvm_asm_scalar_type(cx, s);
            let count = 16 / layout.size.bytes();
            cx.type_vector(elem_ty, count)
        }
        (
            AArch64(AArch64InlineAsmRegClass::vreg_low16),
            BackendRepr::SimdVector { element, count },
        ) if layout.size.bytes() == 8 => {
            let elem_ty = llvm_asm_scalar_type(cx, element);
            cx.type_vector(elem_ty, count * 2)
        }
        (X86(X86InlineAsmRegClass::reg_abcd), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F64) =>
        {
            cx.type_i64()
        }
        (
            X86(X86InlineAsmRegClass::xmm_reg | X86InlineAsmRegClass::zmm_reg),
            BackendRepr::SimdVector { .. },
        ) if layout.size.bytes() == 64 => cx.type_vector(cx.type_f64(), 8),
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::Scalar(s),
        ) if cx.sess().asm_arch == Some(InlineAsmArch::X86)
            && s.primitive() == Primitive::Float(Float::F128) =>
        {
            cx.type_vector(cx.type_i32(), 4)
        }
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::Scalar(s),
        ) if s.primitive() == Primitive::Float(Float::F16) => cx.type_vector(cx.type_i16(), 8),
        (
            X86(
                X86InlineAsmRegClass::xmm_reg
                | X86InlineAsmRegClass::ymm_reg
                | X86InlineAsmRegClass::zmm_reg,
            ),
            BackendRepr::SimdVector { element, count: count @ (8 | 16) },
        ) if element.primitive() == Primitive::Float(Float::F16) => {
            cx.type_vector(cx.type_i16(), count)
        }
        (
            Arm(ArmInlineAsmRegClass::sreg | ArmInlineAsmRegClass::sreg_low16),
            BackendRepr::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I32, _) = s.primitive() {
                cx.type_f32()
            } else {
                layout.llvm_type(cx)
            }
        }
        (
            Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16,
            ),
            BackendRepr::Scalar(s),
        ) => {
            if let Primitive::Int(Integer::I64, _) = s.primitive() {
                cx.type_f64()
            } else {
                layout.llvm_type(cx)
            }
        }
        (
            Arm(
                ArmInlineAsmRegClass::dreg
                | ArmInlineAsmRegClass::dreg_low8
                | ArmInlineAsmRegClass::dreg_low16
                | ArmInlineAsmRegClass::qreg
                | ArmInlineAsmRegClass::qreg_low4
                | ArmInlineAsmRegClass::qreg_low8,
            ),
            BackendRepr::SimdVector { element, count: count @ (4 | 8) },
        ) if element.primitive() == Primitive::Float(Float::F16) => {
            cx.type_vector(cx.type_i16(), count)
        }
        (LoongArch(LoongArchInlineAsmRegClass::freg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F16) =>
        {
            cx.type_f32()
        }
        (Mips(MipsInlineAsmRegClass::reg), BackendRepr::Scalar(s)) => {
            match s.primitive() {
                // MIPS only supports register-length arithmetics.
                Primitive::Int(Integer::I8 | Integer::I16, _) => cx.type_i32(),
                Primitive::Float(Float::F32) => cx.type_i32(),
                Primitive::Float(Float::F64) => cx.type_i64(),
                _ => layout.llvm_type(cx),
            }
        }
        (RiscV(RiscVInlineAsmRegClass::freg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F16)
                && !any_target_feature_enabled(cx, instance, &[sym::zfhmin, sym::zfh]) =>
        {
            cx.type_f32()
        }
        (PowerPC(PowerPCInlineAsmRegClass::vreg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F32) =>
        {
            cx.type_vector(cx.type_f32(), 4)
        }
        (PowerPC(PowerPCInlineAsmRegClass::vreg), BackendRepr::Scalar(s))
            if s.primitive() == Primitive::Float(Float::F64) =>
        {
            cx.type_vector(cx.type_f64(), 2)
        }
        _ => layout.llvm_type(cx),
    }
}
