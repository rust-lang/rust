use gccjit::{LValue, RValue, ToRValue, Type};
use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{
    AsmBuilderMethods, AsmMethods, BaseTypeMethods, BuilderMethods, GlobalAsmOperandRef,
    InlineAsmOperandRef,
};

use rustc_middle::{bug, ty::Instance};
use rustc_span::Span;
use rustc_target::asm::*;

use std::borrow::Cow;

use crate::builder::Builder;
use crate::callee::get_fn;
use crate::context::CodegenCx;
use crate::errors::UnwindingInlineAsm;
use crate::type_of::LayoutGccExt;

// Rust asm! and GCC Extended Asm semantics differ substantially.
//
// 1. Rust asm operands go along as one list of operands. Operands themselves indicate
//    if they're "in" or "out". "In" and "out" operands can interleave. One operand can be
//    both "in" and "out" (`inout(reg)`).
//
//    GCC asm has two different lists for "in" and "out" operands. In terms of gccjit,
//    this means that all "out" operands must go before "in" operands. "In" and "out" operands
//    cannot interleave.
//
// 2. Operand lists in both Rust and GCC are indexed. Index starts from 0. Indexes are important
//    because the asm template refers to operands by index.
//
//    Mapping from Rust to GCC index would be 1-1 if it wasn't for...
//
// 3. Clobbers. GCC has a separate list of clobbers, and clobbers don't have indexes.
//    Contrary, Rust expresses clobbers through "out" operands that aren't tied to
//    a variable (`_`),  and such "clobbers" do have index.
//
// 4. Furthermore, GCC Extended Asm does not support explicit register constraints
//    (like `out("eax")`) directly, offering so-called "local register variables"
//    as a workaround. These variables need to be declared and initialized *before*
//    the Extended Asm block but *after* normal local variables
//    (see comment in `codegen_inline_asm` for explanation).
//
// With that in mind, let's see how we translate Rust syntax to GCC
// (from now on, `CC` stands for "constraint code"):
//
// * `out(reg_class) var`   -> translated to output operand: `"=CC"(var)`
// * `inout(reg_class) var` -> translated to output operand: `"+CC"(var)`
// * `in(reg_class) var`    -> translated to input operand: `"CC"(var)`
//
// * `out(reg_class) _` -> translated to one `=r(tmp)`, where "tmp" is a temporary unused variable
//
// * `out("explicit register") _` -> not translated to any operands, register is simply added to clobbers list
//
// * `inout(reg_class) in_var => out_var` -> translated to two operands:
//                              output: `"=CC"(in_var)`
//                              input:  `"num"(out_var)` where num is the GCC index
//                                       of the corresponding output operand
//
// * `inout(reg_class) in_var => _` -> same as `inout(reg_class) in_var => tmp`,
//                                      where "tmp" is a temporary unused variable
//
// * `out/in/inout("explicit register") var` -> translated to one or two operands as described above
//                                              with `"r"(var)` constraint,
//                                              and one register variable assigned to the desired register.

const ATT_SYNTAX_INS: &str = ".att_syntax noprefix\n\t";
const INTEL_SYNTAX_INS: &str = "\n\t.intel_syntax noprefix";

struct AsmOutOperand<'a, 'tcx, 'gcc> {
    rust_idx: usize,
    constraint: &'a str,
    late: bool,
    readwrite: bool,

    tmp_var: LValue<'gcc>,
    out_place: Option<PlaceRef<'tcx, RValue<'gcc>>>,
}

struct AsmInOperand<'a, 'tcx> {
    rust_idx: usize,
    constraint: Cow<'a, str>,
    val: RValue<'tcx>,
}

impl AsmOutOperand<'_, '_, '_> {
    fn to_constraint(&self) -> String {
        let mut res = String::with_capacity(self.constraint.len() + self.late as usize + 1);

        let sign = if self.readwrite { '+' } else { '=' };
        res.push(sign);
        if !self.late {
            res.push('&');
        }

        res.push_str(self.constraint);
        res
    }
}

enum ConstraintOrRegister {
    Constraint(&'static str),
    Register(&'static str),
}

impl<'a, 'gcc, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        rust_operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        span: &[Span],
        instance: Instance<'_>,
        dest: Option<Self::BasicBlock>,
        _dest_catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
    ) {
        if options.contains(InlineAsmOptions::MAY_UNWIND) {
            self.sess().dcx().create_err(UnwindingInlineAsm { span: span[0] }).emit();
            return;
        }

        let asm_arch = self.tcx.sess.asm_arch.unwrap();
        let is_x86 = matches!(asm_arch, InlineAsmArch::X86 | InlineAsmArch::X86_64);
        let att_dialect = is_x86 && options.contains(InlineAsmOptions::ATT_SYNTAX);

        // GCC index of an output operand equals its position in the array
        let mut outputs = vec![];

        // GCC index of an input operand equals its position in the array
        // added to `outputs.len()`
        let mut inputs = vec![];

        // GCC index of a label equals its position in the array added to
        // `outputs.len() + inputs.len()`.
        let mut labels = vec![];

        // Clobbers collected from `out("explicit register") _` and `inout("expl_reg") var => _`
        let mut clobbers = vec![];

        // We're trying to preallocate space for the template
        let mut constants_len = 0;

        // There are rules we must adhere to if we want GCC to do the right thing:
        //
        // * Every local variable that the asm block uses as an output must be declared *before*
        //   the asm block.
        // * There must be no instructions whatsoever between the register variables and the asm.
        //
        // Therefore, the backend must generate the instructions strictly in this order:
        //
        // 1. Output variables.
        // 2. Register variables.
        // 3. The asm block.
        //
        // We also must make sure that no input operands are emitted before output operands.
        //
        // This is why we work in passes, first emitting local vars, then local register vars.
        // Also, we don't emit any asm operands immediately; we save them to
        // the one of the buffers to be emitted later.

        // 1. Normal variables (and saving operands to buffers).
        for (rust_idx, op) in rust_operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::Out { reg, late, place } => {
                    use ConstraintOrRegister::*;

                    let (constraint, ty) = match (reg_to_gcc(reg), place) {
                        (Constraint(constraint), Some(place)) => {
                            (constraint, place.layout.gcc_type(self.cx))
                        }
                        // When `reg` is a class and not an explicit register but the out place is not specified,
                        // we need to create an unused output variable to assign the output to. This var
                        // needs to be of a type that's "compatible" with the register class, but specific type
                        // doesn't matter.
                        (Constraint(constraint), None) => {
                            (constraint, dummy_output_type(self.cx, reg.reg_class()))
                        }
                        (Register(_), Some(_)) => {
                            // left for the next pass
                            continue;
                        }
                        (Register(reg_name), None) => {
                            // `clobber_abi` can add lots of clobbers that are not supported by the target,
                            // such as AVX-512 registers, so we just ignore unsupported registers
                            let is_target_supported =
                                reg.reg_class().supported_types(asm_arch).iter().any(
                                    |&(_, feature)| {
                                        if let Some(feature) = feature {
                                            self.tcx
                                                .asm_target_features(instance.def_id())
                                                .contains(&feature)
                                        } else {
                                            true // Register class is unconditionally supported
                                        }
                                    },
                                );

                            if is_target_supported && !clobbers.contains(&reg_name) {
                                clobbers.push(reg_name);
                            }
                            continue;
                        }
                    };

                    let tmp_var = self.current_func().new_local(None, ty, "output_register");
                    outputs.push(AsmOutOperand {
                        constraint,
                        rust_idx,
                        late,
                        readwrite: false,
                        tmp_var,
                        out_place: place,
                    });
                }

                InlineAsmOperandRef::In { reg, value } => {
                    if let ConstraintOrRegister::Constraint(constraint) = reg_to_gcc(reg) {
                        inputs.push(AsmInOperand {
                            constraint: Cow::Borrowed(constraint),
                            rust_idx,
                            val: value.immediate(),
                        });
                    } else {
                        // left for the next pass
                        continue;
                    }
                }

                InlineAsmOperandRef::InOut { reg, late, in_value, out_place } => {
                    let constraint =
                        if let ConstraintOrRegister::Constraint(constraint) = reg_to_gcc(reg) {
                            constraint
                        } else {
                            // left for the next pass
                            continue;
                        };

                    // Rustc frontend guarantees that input and output types are "compatible",
                    // so we can just use input var's type for the output variable.
                    //
                    // This decision is also backed by the fact that LLVM needs in and out
                    // values to be of *exactly the same type*, not just "compatible".
                    // I'm not sure if GCC is so picky too, but better safe than sorry.
                    let ty = in_value.layout.gcc_type(self.cx);
                    let tmp_var = self.current_func().new_local(None, ty, "output_register");

                    // If the out_place is None (i.e `inout(reg) _` syntax was used), we translate
                    // it to one "readwrite (+) output variable", otherwise we translate it to two
                    // "out and tied in" vars as described above.
                    let readwrite = out_place.is_none();
                    outputs.push(AsmOutOperand {
                        constraint,
                        rust_idx,
                        late,
                        readwrite,
                        tmp_var,
                        out_place,
                    });

                    if !readwrite {
                        let out_gcc_idx = outputs.len() - 1;
                        let constraint = Cow::Owned(out_gcc_idx.to_string());

                        inputs.push(AsmInOperand {
                            constraint,
                            rust_idx,
                            val: in_value.immediate(),
                        });
                    }
                }

                InlineAsmOperandRef::Const { ref string } => {
                    constants_len += string.len() + att_dialect as usize;
                }

                InlineAsmOperandRef::SymFn { instance } => {
                    // TODO(@Amanieu): Additional mangling is needed on
                    // some targets to add a leading underscore (Mach-O)
                    // or byte count suffixes (x86 Windows).
                    constants_len += self.tcx.symbol_name(instance).name.len();
                }
                InlineAsmOperandRef::SymStatic { def_id } => {
                    // TODO(@Amanieu): Additional mangling is needed on
                    // some targets to add a leading underscore (Mach-O).
                    constants_len +=
                        self.tcx.symbol_name(Instance::mono(self.tcx, def_id)).name.len();
                }

                InlineAsmOperandRef::Label { label } => {
                    labels.push(label);
                }
            }
        }

        // 2. Register variables.
        for (rust_idx, op) in rust_operands.iter().enumerate() {
            match *op {
                // `out("explicit register") var`
                InlineAsmOperandRef::Out { reg, late, place } => {
                    if let ConstraintOrRegister::Register(reg_name) = reg_to_gcc(reg) {
                        let out_place = if let Some(place) = place {
                            place
                        } else {
                            // processed in the previous pass
                            continue;
                        };

                        let ty = out_place.layout.gcc_type(self.cx);
                        let tmp_var = self.current_func().new_local(None, ty, "output_register");
                        tmp_var.set_register_name(reg_name);

                        outputs.push(AsmOutOperand {
                            constraint: "r",
                            rust_idx,
                            late,
                            readwrite: false,
                            tmp_var,
                            out_place: Some(out_place),
                        });
                    }

                    // processed in the previous pass
                }

                // `in("explicit register") var`
                InlineAsmOperandRef::In { reg, value } => {
                    if let ConstraintOrRegister::Register(reg_name) = reg_to_gcc(reg) {
                        let ty = value.layout.gcc_type(self.cx);
                        let reg_var = self.current_func().new_local(None, ty, "input_register");
                        reg_var.set_register_name(reg_name);
                        self.llbb().add_assignment(None, reg_var, value.immediate());

                        inputs.push(AsmInOperand {
                            constraint: "r".into(),
                            rust_idx,
                            val: reg_var.to_rvalue(),
                        });
                    }

                    // processed in the previous pass
                }

                // `inout("explicit register") in_var => out_var`
                InlineAsmOperandRef::InOut { reg, late, in_value, out_place } => {
                    if let ConstraintOrRegister::Register(reg_name) = reg_to_gcc(reg) {
                        // See explanation in the first pass.
                        let ty = in_value.layout.gcc_type(self.cx);
                        let tmp_var = self.current_func().new_local(None, ty, "output_register");
                        tmp_var.set_register_name(reg_name);

                        outputs.push(AsmOutOperand {
                            constraint: "r",
                            rust_idx,
                            late,
                            readwrite: false,
                            tmp_var,
                            out_place,
                        });

                        let constraint = Cow::Owned((outputs.len() - 1).to_string());
                        inputs.push(AsmInOperand {
                            constraint,
                            rust_idx,
                            val: in_value.immediate(),
                        });
                    }

                    // processed in the previous pass
                }

                InlineAsmOperandRef::SymFn { instance } => {
                    inputs.push(AsmInOperand {
                        constraint: "X".into(),
                        rust_idx,
                        val: get_fn(self.cx, instance).get_address(None),
                    });
                }

                InlineAsmOperandRef::SymStatic { def_id } => {
                    inputs.push(AsmInOperand {
                        constraint: "X".into(),
                        rust_idx,
                        val: self.cx.get_static(def_id).get_address(None),
                    });
                }

                InlineAsmOperandRef::Const { .. } => {
                    // processed in the previous pass
                }

                InlineAsmOperandRef::Label { .. } => {
                    // processed in the previous pass
                }
            }
        }

        // 3. Build the template string

        let mut template_str =
            String::with_capacity(estimate_template_length(template, constants_len, att_dialect));
        if att_dialect {
            template_str.push_str(ATT_SYNTAX_INS);
        }

        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref string) => {
                    for char in string.chars() {
                        // TODO(antoyo): might also need to escape | if rustc doesn't do it.
                        let escaped_char = match char {
                            '%' => "%%",
                            '{' => "%{",
                            '}' => "%}",
                            _ => {
                                template_str.push(char);
                                continue;
                            }
                        };
                        template_str.push_str(escaped_char);
                    }
                }
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier, span: _ } => {
                    let mut push_to_template = |modifier, gcc_idx| {
                        use std::fmt::Write;

                        template_str.push('%');
                        if let Some(modifier) = modifier {
                            template_str.push(modifier);
                        }
                        write!(template_str, "{}", gcc_idx).expect("pushing to string failed");
                    };

                    match rust_operands[operand_idx] {
                        InlineAsmOperandRef::Out { reg, .. } => {
                            let modifier = modifier_to_gcc(asm_arch, reg.reg_class(), modifier);
                            let gcc_index = outputs
                                .iter()
                                .position(|op| operand_idx == op.rust_idx)
                                .expect("wrong rust index");
                            push_to_template(modifier, gcc_index);
                        }

                        InlineAsmOperandRef::In { reg, .. } => {
                            let modifier = modifier_to_gcc(asm_arch, reg.reg_class(), modifier);
                            let in_gcc_index = inputs
                                .iter()
                                .position(|op| operand_idx == op.rust_idx)
                                .expect("wrong rust index");
                            let gcc_index = in_gcc_index + outputs.len();
                            push_to_template(modifier, gcc_index);
                        }

                        InlineAsmOperandRef::InOut { reg, .. } => {
                            let modifier = modifier_to_gcc(asm_arch, reg.reg_class(), modifier);

                            // The input register is tied to the output, so we can just use the index of the output register
                            let gcc_index = outputs
                                .iter()
                                .position(|op| operand_idx == op.rust_idx)
                                .expect("wrong rust index");
                            push_to_template(modifier, gcc_index);
                        }

                        InlineAsmOperandRef::SymFn { instance } => {
                            // TODO(@Amanieu): Additional mangling is needed on
                            // some targets to add a leading underscore (Mach-O)
                            // or byte count suffixes (x86 Windows).
                            let name = self.tcx.symbol_name(instance).name;
                            template_str.push_str(name);
                        }

                        InlineAsmOperandRef::SymStatic { def_id } => {
                            // TODO(@Amanieu): Additional mangling is needed on
                            // some targets to add a leading underscore (Mach-O).
                            let instance = Instance::mono(self.tcx, def_id);
                            let name = self.tcx.symbol_name(instance).name;
                            template_str.push_str(name);
                        }

                        InlineAsmOperandRef::Const { ref string } => {
                            template_str.push_str(string);
                        }

                        InlineAsmOperandRef::Label { label } => {
                            let label_gcc_index =
                                labels.iter().position(|&l| l == label).expect("wrong rust index");
                            let gcc_index = label_gcc_index + outputs.len() + inputs.len();
                            push_to_template(Some('l'), gcc_index);
                        }
                    }
                }
            }
        }

        if att_dialect {
            template_str.push_str(INTEL_SYNTAX_INS);
        }

        // 4. Generate Extended Asm block

        let block = self.llbb();
        let extended_asm = if let Some(dest) = dest {
            assert!(!labels.is_empty());
            block.end_with_extended_asm_goto(None, &template_str, &labels, Some(dest))
        } else {
            block.add_extended_asm(None, &template_str)
        };

        for op in &outputs {
            extended_asm.add_output_operand(None, &op.to_constraint(), op.tmp_var);
        }

        for op in &inputs {
            extended_asm.add_input_operand(None, &op.constraint, op.val);
        }

        for clobber in clobbers.iter() {
            extended_asm.add_clobber(clobber);
        }

        if !options.contains(InlineAsmOptions::PRESERVES_FLAGS) {
            // TODO(@Commeownist): I'm not 100% sure this one clobber is sufficient
            // on all architectures. For instance, what about FP stack?
            extended_asm.add_clobber("cc");
        }
        if !options.contains(InlineAsmOptions::NOMEM) {
            extended_asm.add_clobber("memory");
        }
        if !options.contains(InlineAsmOptions::PURE) {
            extended_asm.set_volatile_flag(true);
        }
        if !options.contains(InlineAsmOptions::NOSTACK) {
            // TODO(@Commeownist): figure out how to align stack
        }
        if dest.is_none() && options.contains(InlineAsmOptions::NORETURN) {
            let builtin_unreachable = self.context.get_builtin_function("__builtin_unreachable");
            let builtin_unreachable: RValue<'gcc> =
                unsafe { std::mem::transmute(builtin_unreachable) };
            self.call(self.type_void(), None, None, builtin_unreachable, &[], None, None);
        }

        // Write results to outputs.
        //
        // We need to do this because:
        //  1. Turning `PlaceRef` into `RValue` is error-prone and has nasty edge cases
        //     (especially with current `rustc_backend_ssa` API).
        //  2. Not every output operand has an `out_place`, and it's required by `add_output_operand`.
        //
        // Instead, we generate a temporary output variable for each output operand, and then this loop,
        // generates `out_place = tmp_var;` assignments if out_place exists.
        for op in &outputs {
            if let Some(place) = op.out_place {
                OperandValue::Immediate(op.tmp_var.to_rvalue()).store(self, place);
            }
        }
    }
}

fn estimate_template_length(
    template: &[InlineAsmTemplatePiece],
    constants_len: usize,
    att_dialect: bool,
) -> usize {
    let len: usize = template
        .iter()
        .map(|piece| {
            match *piece {
                InlineAsmTemplatePiece::String(ref string) => string.len(),
                InlineAsmTemplatePiece::Placeholder { .. } => {
                    // '%' + 1 char modifier + 1 char index
                    3
                }
            }
        })
        .sum();

    // increase it by 5% to account for possible '%' signs that'll be duplicated
    // I pulled the number out of blue, but should be fair enough
    // as the upper bound
    let mut res = (len as f32 * 1.05) as usize + constants_len;

    if att_dialect {
        res += INTEL_SYNTAX_INS.len() + ATT_SYNTAX_INS.len();
    }
    res
}

/// Converts a register class to a GCC constraint code.
fn reg_to_gcc(reg: InlineAsmRegOrRegClass) -> ConstraintOrRegister {
    let constraint = match reg {
        // For vector registers LLVM wants the register name to match the type size.
        InlineAsmRegOrRegClass::Reg(reg) => {
            match reg {
                InlineAsmReg::X86(_) => {
                    // TODO(antoyo): add support for vector register.
                    //
                    // // For explicit registers, we have to create a register variable: https://stackoverflow.com/a/31774784/389119
                    return ConstraintOrRegister::Register(match reg.name() {
                        // Some of registers' names does not map 1-1 from rust to gcc
                        "st(0)" => "st",

                        name => name,
                    });
                }

                _ => unimplemented!(),
            }
        }
        // They can be retrieved from https://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html
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
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low8)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg_low16)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low8)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg_low4)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg)
            | InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg) => "t",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_upper) => "d",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_pair) => "r",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_iw) => "w",
            InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_ptr) => "e",
            InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::wreg) => "w",
            InlineAsmRegClass::Hexagon(HexagonInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg_addr) => "a",
            InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg_data) => "d",
            InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg) => "d", // more specific than "r"
            InlineAsmRegClass::Mips(MipsInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::Msp430(Msp430InlineAsmRegClass::reg) => "r",
            // https://github.com/gcc-mirror/gcc/blob/master/gcc/config/nvptx/nvptx.md -> look for
            // "define_constraint".
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
            InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg) => "Yk",
            InlineAsmRegClass::X86(
                X86InlineAsmRegClass::kreg0
                | X86InlineAsmRegClass::x87_reg
                | X86InlineAsmRegClass::mmx_reg
                | X86InlineAsmRegClass::tmm_reg,
            ) => unreachable!("clobber-only"),
            InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg) => {
                bug!("GCC backend does not support SPIR-V")
            }
            InlineAsmRegClass::Wasm(WasmInlineAsmRegClass::local) => "r",
            InlineAsmRegClass::S390x(S390xInlineAsmRegClass::reg) => "r",
            InlineAsmRegClass::S390x(S390xInlineAsmRegClass::reg_addr) => "a",
            InlineAsmRegClass::S390x(S390xInlineAsmRegClass::freg) => "f",
            InlineAsmRegClass::Err => unreachable!(),
        },
    };

    ConstraintOrRegister::Constraint(constraint)
}

/// Type to use for outputs that are discarded. It doesn't really matter what
/// the type is, as long as it is valid for the constraint code.
fn dummy_output_type<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, reg: InlineAsmRegClass) -> Type<'gcc> {
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
        InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::freg) => cx.type_f32(),
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
        InlineAsmRegClass::X86(X86InlineAsmRegClass::x87_reg)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::mmx_reg)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg0)
        | InlineAsmRegClass::X86(X86InlineAsmRegClass::tmm_reg) => {
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
        InlineAsmRegClass::S390x(
            S390xInlineAsmRegClass::reg | S390xInlineAsmRegClass::reg_addr,
        ) => cx.type_i32(),
        InlineAsmRegClass::S390x(S390xInlineAsmRegClass::freg) => cx.type_f64(),
        InlineAsmRegClass::Msp430(Msp430InlineAsmRegClass::reg) => cx.type_i16(),
        InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg_addr) => cx.type_i32(),
        InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg_data) => cx.type_i32(),
        InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::reg) => cx.type_i32(),
        InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::freg) => cx.type_f32(),
        InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg) => {
            bug!("GCC backend does not support SPIR-V")
        }
        InlineAsmRegClass::Err => unreachable!(),
    }
}

impl<'gcc, 'tcx> AsmMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn codegen_global_asm(
        &self,
        template: &[InlineAsmTemplatePiece],
        operands: &[GlobalAsmOperandRef<'tcx>],
        options: InlineAsmOptions,
        _line_spans: &[Span],
    ) {
        let asm_arch = self.tcx.sess.asm_arch.unwrap();

        // Default to Intel syntax on x86
        let att_dialect = matches!(asm_arch, InlineAsmArch::X86 | InlineAsmArch::X86_64)
            && options.contains(InlineAsmOptions::ATT_SYNTAX);

        // Build the template string
        let mut template_str = ".pushsection .text\n".to_owned();
        if att_dialect {
            template_str.push_str(".att_syntax\n");
        }
        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref string) => {
                    let mut index = 0;
                    while index < string.len() {
                        // NOTE: gcc does not allow inline comment, so remove them.
                        let comment_index = string[index..]
                            .find("//")
                            .map(|comment_index| comment_index + index)
                            .unwrap_or(string.len());
                        template_str.push_str(&string[index..comment_index]);
                        index = string[comment_index..]
                            .find('\n')
                            .map(|index| index + comment_index)
                            .unwrap_or(string.len());
                    }
                }
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier: _, span: _ } => {
                    match operands[operand_idx] {
                        GlobalAsmOperandRef::Const { ref string } => {
                            // Const operands get injected directly into the
                            // template. Note that we don't need to escape %
                            // here unlike normal inline assembly.
                            template_str.push_str(string);
                        }

                        GlobalAsmOperandRef::SymFn { instance } => {
                            let function = get_fn(self, instance);
                            self.add_used_function(function);
                            // TODO(@Amanieu): Additional mangling is needed on
                            // some targets to add a leading underscore (Mach-O)
                            // or byte count suffixes (x86 Windows).
                            let name = self.tcx.symbol_name(instance).name;
                            template_str.push_str(name);
                        }

                        GlobalAsmOperandRef::SymStatic { def_id } => {
                            // TODO(antoyo): set the global variable as used.
                            // TODO(@Amanieu): Additional mangling is needed on
                            // some targets to add a leading underscore (Mach-O).
                            let instance = Instance::mono(self.tcx, def_id);
                            let name = self.tcx.symbol_name(instance).name;
                            template_str.push_str(name);
                        }
                    }
                }
            }
        }

        if att_dialect {
            template_str.push_str("\n\t.intel_syntax noprefix");
        }
        // NOTE: seems like gcc will put the asm in the wrong section, so set it to .text manually.
        template_str.push_str("\n.popsection");
        self.context.add_top_level_asm(None, &template_str);
    }
}

fn modifier_to_gcc(
    arch: InlineAsmArch,
    reg: InlineAsmRegClass,
    modifier: Option<char>,
) -> Option<char> {
    // The modifiers can be retrieved from
    // https://gcc.gnu.org/onlinedocs/gcc/Modifiers.html#Modifiers
    match reg {
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::reg) => modifier,
        InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg)
        | InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16) => {
            if modifier == Some('v') {
                None
            } else {
                modifier
            }
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
        InlineAsmRegClass::LoongArch(_) => None,
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
            None => {
                if arch == InlineAsmArch::X86_64 {
                    Some('q')
                } else {
                    Some('k')
                }
            }
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
        InlineAsmRegClass::M68k(_) => None,
        InlineAsmRegClass::CSKY(_) => None,
        InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg) => {
            bug!("LLVM backend does not support SPIR-V")
        }
        InlineAsmRegClass::Err => unreachable!(),
    }
}
