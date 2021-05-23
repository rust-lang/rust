use super::LoweringContext;

use rustc_ast::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_span::{Span, Symbol};
use rustc_target::asm;
use std::collections::hash_map::Entry;
use std::fmt::Write;

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    crate fn lower_inline_asm(&mut self, sp: Span, asm: &InlineAsm) -> &'hir hir::InlineAsm<'hir> {
        // Rustdoc needs to support asm! from foriegn architectures: don't try
        // lowering the register contraints in this case.
        let asm_arch = if self.sess.opts.actually_rustdoc { None } else { self.sess.asm_arch };
        if asm_arch.is_none() && !self.sess.opts.actually_rustdoc {
            struct_span_err!(self.sess, sp, E0472, "inline assembly is unsupported on this target")
                .emit();
        }
        if asm.options.contains(InlineAsmOptions::ATT_SYNTAX)
            && !matches!(asm_arch, Some(asm::InlineAsmArch::X86 | asm::InlineAsmArch::X86_64))
            && !self.sess.opts.actually_rustdoc
        {
            self.sess
                .struct_span_err(sp, "the `att_syntax` option is only supported on x86")
                .emit();
        }

        // Lower operands to HIR. We use dummy register classes if an error
        // occurs during lowering because we still need to be able to produce a
        // valid HIR.
        let sess = self.sess;
        let operands: Vec<_> = asm
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
                    InlineAsmOperand::Const { ref anon_const } => hir::InlineAsmOperand::Const {
                        anon_const: self.lower_anon_const(anon_const),
                    },
                    InlineAsmOperand::Sym { ref expr } => {
                        hir::InlineAsmOperand::Sym { expr: self.lower_expr_mut(expr) }
                    }
                };
                (op, *op_sp)
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
        let mut required_features: Vec<&str> = vec![];
        for (idx, &(ref op, op_sp)) in operands.iter().enumerate() {
            if let Some(reg) = op.reg() {
                // Make sure we don't accidentally carry features from the
                // previous iteration.
                required_features.clear();

                let reg_class = reg.reg_class();
                if reg_class == asm::InlineAsmRegClass::Err {
                    continue;
                }

                // We ignore target feature requirements for clobbers: if the
                // feature is disabled then the compiler doesn't care what we
                // do with the registers.
                //
                // Note that this is only possible for explicit register
                // operands, which cannot be used in the asm string.
                let is_clobber = matches!(
                    op,
                    hir::InlineAsmOperand::Out {
                        reg: asm::InlineAsmRegOrRegClass::Reg(_),
                        late: _,
                        expr: None
                    }
                );

                if !is_clobber {
                    // Validate register classes against currently enabled target
                    // features. We check that at least one type is available for
                    // the current target.
                    for &(_, feature) in reg_class.supported_types(asm_arch.unwrap()) {
                        if let Some(feature) = feature {
                            if self.sess.target_features.contains(&Symbol::intern(feature)) {
                                required_features.clear();
                                break;
                            } else {
                                required_features.push(feature);
                            }
                        } else {
                            required_features.clear();
                            break;
                        }
                    }
                    // We are sorting primitive strs here and can use unstable sort here
                    required_features.sort_unstable();
                    required_features.dedup();
                    match &required_features[..] {
                        [] => {}
                        [feature] => {
                            let msg = format!(
                                "register class `{}` requires the `{}` target feature",
                                reg_class.name(),
                                feature
                            );
                            sess.struct_span_err(op_sp, &msg).emit();
                        }
                        features => {
                            let msg = format!(
                                "register class `{}` requires at least one target feature: {}",
                                reg_class.name(),
                                features.join(", ")
                            );
                            sess.struct_span_err(op_sp, &msg).emit();
                        }
                    }
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

        let operands = self.arena.alloc_from_iter(operands);
        let template = self.arena.alloc_from_iter(asm.template.iter().cloned());
        let line_spans = self.arena.alloc_slice(&asm.line_spans[..]);
        let hir_asm = hir::InlineAsm { template, operands, options: asm.options, line_spans };
        self.arena.alloc(hir_asm)
    }
}
