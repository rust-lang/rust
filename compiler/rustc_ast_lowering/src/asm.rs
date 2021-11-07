use super::LoweringContext;

use rustc_ast::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_session::parse::feature_err;
use rustc_span::{sym, Span, Symbol};
use rustc_target::asm;
use std::collections::hash_map::Entry;
use std::fmt::Write;

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    crate fn lower_inline_asm(&mut self, sp: Span, asm: &InlineAsm) -> &'hir hir::InlineAsm<'hir> {
        // Rustdoc needs to support asm! from foreign architectures: don't try
        // lowering the register constraints in this case.
        let asm_arch = if self.sess.opts.actually_rustdoc { None } else { self.sess.asm_arch };
        if asm_arch.is_none() && !self.sess.opts.actually_rustdoc {
            struct_span_err!(self.sess, sp, E0472, "inline assembly is unsupported on this target")
                .emit();
        }
        if let Some(asm_arch) = asm_arch {
            // Inline assembly is currently only stable for these architectures.
            let is_stable = matches!(
                asm_arch,
                asm::InlineAsmArch::X86
                    | asm::InlineAsmArch::X86_64
                    | asm::InlineAsmArch::Arm
                    | asm::InlineAsmArch::AArch64
                    | asm::InlineAsmArch::RiscV32
                    | asm::InlineAsmArch::RiscV64
            );
            if !is_stable && !self.sess.features_untracked().asm_experimental_arch {
                feature_err(
                    &self.sess.parse_sess,
                    sym::asm_experimental_arch,
                    sp,
                    "inline assembly is not stable yet on this architecture",
                )
                .emit();
            }
        }
        if asm.options.contains(InlineAsmOptions::ATT_SYNTAX)
            && !matches!(asm_arch, Some(asm::InlineAsmArch::X86 | asm::InlineAsmArch::X86_64))
            && !self.sess.opts.actually_rustdoc
        {
            self.sess
                .struct_span_err(sp, "the `att_syntax` option is only supported on x86")
                .emit();
        }

        let mut clobber_abi = None;
        if let Some(asm_arch) = asm_arch {
            if let Some((abi_name, abi_span)) = asm.clobber_abi {
                match asm::InlineAsmClobberAbi::parse(asm_arch, &self.sess.target, abi_name) {
                    Ok(abi) => clobber_abi = Some((abi, abi_span)),
                    Err(&[]) => {
                        self.sess
                            .struct_span_err(
                                abi_span,
                                "`clobber_abi` is not supported on this target",
                            )
                            .emit();
                    }
                    Err(supported_abis) => {
                        let mut err =
                            self.sess.struct_span_err(abi_span, "invalid ABI for `clobber_abi`");
                        let mut abis = format!("`{}`", supported_abis[0]);
                        for m in &supported_abis[1..] {
                            let _ = write!(abis, ", `{}`", m);
                        }
                        err.note(&format!(
                            "the following ABIs are supported on this target: {}",
                            abis
                        ));
                        err.emit();
                    }
                }
            }
        }

        // Lower operands to HIR. We use dummy register classes if an error
        // occurs during lowering because we still need to be able to produce a
        // valid HIR.
        let sess = self.sess;
        let mut operands: Vec<_> = asm
            .operands
            .iter()
            .map(|(op, op_sp)| {
                let lower_reg = |reg| match reg {
                    InlineAsmRegOrRegClass::Reg(s) => {
                        asm::InlineAsmRegOrRegClass::Reg(if let Some(asm_arch) = asm_arch {
                            asm::InlineAsmReg::parse(
                                asm_arch,
                                |feature| sess.target_features.contains(&Symbol::intern(feature)),
                                &sess.target,
                                s,
                            )
                            .unwrap_or_else(|e| {
                                let msg = format!("invalid register `{}`: {}", s.as_str(), e);
                                sess.struct_span_err(*op_sp, &msg).emit();
                                asm::InlineAsmReg::Err
                            })
                        } else {
                            asm::InlineAsmReg::Err
                        })
                    }
                    InlineAsmRegOrRegClass::RegClass(s) => {
                        asm::InlineAsmRegOrRegClass::RegClass(if let Some(asm_arch) = asm_arch {
                            asm::InlineAsmRegClass::parse(asm_arch, s).unwrap_or_else(|e| {
                                let msg = format!("invalid register class `{}`: {}", s.as_str(), e);
                                sess.struct_span_err(*op_sp, &msg).emit();
                                asm::InlineAsmRegClass::Err
                            })
                        } else {
                            asm::InlineAsmRegClass::Err
                        })
                    }
                };

                let op = match *op {
                    InlineAsmOperand::In { reg, ref expr } => hir::InlineAsmOperand::In {
                        reg: lower_reg(reg),
                        expr: self.lower_expr_mut(expr),
                    },
                    InlineAsmOperand::Out { reg, late, ref expr } => hir::InlineAsmOperand::Out {
                        reg: lower_reg(reg),
                        late,
                        expr: expr.as_ref().map(|expr| self.lower_expr_mut(expr)),
                    },
                    InlineAsmOperand::InOut { reg, late, ref expr } => {
                        hir::InlineAsmOperand::InOut {
                            reg: lower_reg(reg),
                            late,
                            expr: self.lower_expr_mut(expr),
                        }
                    }
                    InlineAsmOperand::SplitInOut { reg, late, ref in_expr, ref out_expr } => {
                        hir::InlineAsmOperand::SplitInOut {
                            reg: lower_reg(reg),
                            late,
                            in_expr: self.lower_expr_mut(in_expr),
                            out_expr: out_expr.as_ref().map(|expr| self.lower_expr_mut(expr)),
                        }
                    }
                    InlineAsmOperand::Const { ref anon_const } => {
                        if !self.sess.features_untracked().asm_const {
                            feature_err(
                                &self.sess.parse_sess,
                                sym::asm_const,
                                *op_sp,
                                "const operands for inline assembly are unstable",
                            )
                            .emit();
                        }
                        hir::InlineAsmOperand::Const {
                            anon_const: self.lower_anon_const(anon_const),
                        }
                    }
                    InlineAsmOperand::Sym { ref expr } => {
                        if !self.sess.features_untracked().asm_sym {
                            feature_err(
                                &self.sess.parse_sess,
                                sym::asm_sym,
                                *op_sp,
                                "sym operands for inline assembly are unstable",
                            )
                            .emit();
                        }
                        hir::InlineAsmOperand::Sym { expr: self.lower_expr_mut(expr) }
                    }
                };
                (op, self.lower_span(*op_sp))
            })
            .collect();

        // Validate template modifiers against the register classes for the operands
        for p in &asm.template {
            if let InlineAsmTemplatePiece::Placeholder {
                operand_idx,
                modifier: Some(modifier),
                span: placeholder_span,
            } = *p
            {
                let op_sp = asm.operands[operand_idx].1;
                match &operands[operand_idx].0 {
                    hir::InlineAsmOperand::In { reg, .. }
                    | hir::InlineAsmOperand::Out { reg, .. }
                    | hir::InlineAsmOperand::InOut { reg, .. }
                    | hir::InlineAsmOperand::SplitInOut { reg, .. } => {
                        let class = reg.reg_class();
                        if class == asm::InlineAsmRegClass::Err {
                            continue;
                        }
                        let valid_modifiers = class.valid_modifiers(asm_arch.unwrap());
                        if !valid_modifiers.contains(&modifier) {
                            let mut err = sess.struct_span_err(
                                placeholder_span,
                                "invalid asm template modifier for this register class",
                            );
                            err.span_label(placeholder_span, "template modifier");
                            err.span_label(op_sp, "argument");
                            if !valid_modifiers.is_empty() {
                                let mut mods = format!("`{}`", valid_modifiers[0]);
                                for m in &valid_modifiers[1..] {
                                    let _ = write!(mods, ", `{}`", m);
                                }
                                err.note(&format!(
                                    "the `{}` register class supports \
                                     the following template modifiers: {}",
                                    class.name(),
                                    mods
                                ));
                            } else {
                                err.note(&format!(
                                    "the `{}` register class does not support template modifiers",
                                    class.name()
                                ));
                            }
                            err.emit();
                        }
                    }
                    hir::InlineAsmOperand::Const { .. } => {
                        let mut err = sess.struct_span_err(
                            placeholder_span,
                            "asm template modifiers are not allowed for `const` arguments",
                        );
                        err.span_label(placeholder_span, "template modifier");
                        err.span_label(op_sp, "argument");
                        err.emit();
                    }
                    hir::InlineAsmOperand::Sym { .. } => {
                        let mut err = sess.struct_span_err(
                            placeholder_span,
                            "asm template modifiers are not allowed for `sym` arguments",
                        );
                        err.span_label(placeholder_span, "template modifier");
                        err.span_label(op_sp, "argument");
                        err.emit();
                    }
                }
            }
        }

        let mut used_input_regs = FxHashMap::default();
        let mut used_output_regs = FxHashMap::default();

        for (idx, &(ref op, op_sp)) in operands.iter().enumerate() {
            if let Some(reg) = op.reg() {
                let reg_class = reg.reg_class();
                if reg_class == asm::InlineAsmRegClass::Err {
                    continue;
                }

                // Some register classes can only be used as clobbers. This
                // means that we disallow passing a value in/out of the asm and
                // require that the operand name an explicit register, not a
                // register class.
                if reg_class.is_clobber_only(asm_arch.unwrap()) && !op.is_clobber() {
                    let msg = format!(
                        "register class `{}` can only be used as a clobber, \
                             not as an input or output",
                        reg_class.name()
                    );
                    sess.struct_span_err(op_sp, &msg).emit();
                    continue;
                }

                // Check for conflicts between explicit register operands.
                if let asm::InlineAsmRegOrRegClass::Reg(reg) = reg {
                    let (input, output) = match op {
                        hir::InlineAsmOperand::In { .. } => (true, false),

                        // Late output do not conflict with inputs, but normal outputs do
                        hir::InlineAsmOperand::Out { late, .. } => (!late, true),

                        hir::InlineAsmOperand::InOut { .. }
                        | hir::InlineAsmOperand::SplitInOut { .. } => (true, true),

                        hir::InlineAsmOperand::Const { .. } | hir::InlineAsmOperand::Sym { .. } => {
                            unreachable!()
                        }
                    };

                    // Flag to output the error only once per operand
                    let mut skip = false;
                    reg.overlapping_regs(|r| {
                        let mut check = |used_regs: &mut FxHashMap<asm::InlineAsmReg, usize>,
                                         input| {
                            match used_regs.entry(r) {
                                Entry::Occupied(o) => {
                                    if skip {
                                        return;
                                    }
                                    skip = true;

                                    let idx2 = *o.get();
                                    let &(ref op2, op_sp2) = &operands[idx2];
                                    let reg2 = match op2.reg() {
                                        Some(asm::InlineAsmRegOrRegClass::Reg(r)) => r,
                                        _ => unreachable!(),
                                    };

                                    let msg = format!(
                                        "register `{}` conflicts with register `{}`",
                                        reg.name(),
                                        reg2.name()
                                    );
                                    let mut err = sess.struct_span_err(op_sp, &msg);
                                    err.span_label(op_sp, &format!("register `{}`", reg.name()));
                                    err.span_label(op_sp2, &format!("register `{}`", reg2.name()));

                                    match (op, op2) {
                                        (
                                            hir::InlineAsmOperand::In { .. },
                                            hir::InlineAsmOperand::Out { late, .. },
                                        )
                                        | (
                                            hir::InlineAsmOperand::Out { late, .. },
                                            hir::InlineAsmOperand::In { .. },
                                        ) => {
                                            assert!(!*late);
                                            let out_op_sp = if input { op_sp2 } else { op_sp };
                                            let msg = "use `lateout` instead of \
                                                       `out` to avoid conflict";
                                            err.span_help(out_op_sp, msg);
                                        }
                                        _ => {}
                                    }

                                    err.emit();
                                }
                                Entry::Vacant(v) => {
                                    v.insert(idx);
                                }
                            }
                        };
                        if input {
                            check(&mut used_input_regs, true);
                        }
                        if output {
                            check(&mut used_output_regs, false);
                        }
                    });
                }
            }
        }

        // If a clobber_abi is specified, add the necessary clobbers to the
        // operands list.
        if let Some((abi, abi_span)) = clobber_abi {
            for &clobber in abi.clobbered_regs() {
                let mut output_used = false;
                clobber.overlapping_regs(|reg| {
                    if used_output_regs.contains_key(&reg) {
                        output_used = true;
                    }
                });

                if !output_used {
                    operands.push((
                        hir::InlineAsmOperand::Out {
                            reg: asm::InlineAsmRegOrRegClass::Reg(clobber),
                            late: true,
                            expr: None,
                        },
                        self.lower_span(abi_span),
                    ));
                }
            }
        }

        let operands = self.arena.alloc_from_iter(operands);
        let template = self.arena.alloc_from_iter(asm.template.iter().cloned());
        let template_strs = self.arena.alloc_from_iter(
            asm.template_strs
                .iter()
                .map(|(sym, snippet, span)| (*sym, *snippet, self.lower_span(*span))),
        );
        let line_spans =
            self.arena.alloc_from_iter(asm.line_spans.iter().map(|span| self.lower_span(*span)));
        let hir_asm =
            hir::InlineAsm { template, template_strs, operands, options: asm.options, line_spans };
        self.arena.alloc(hir_asm)
    }
}
