//! Emulate AArch64 LLVM intrinsics

use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_target::asm::*;

use crate::inline_asm::{CInlineAsmOperand, codegen_inline_asm_inner};
use crate::intrinsics::*;
use crate::prelude::*;

pub(super) fn codegen_aarch64_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    args: &[Spanned<mir::Operand<'tcx>>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
) {
    // llvm.aarch64.neon.sqshl.v*i*

    match intrinsic {
        "llvm.aarch64.isb" => {
            fx.bcx.ins().fence();
        }

        "llvm.aarch64.neon.ld1x4.v16i8.p0" => {
            intrinsic_args!(fx, args => (ptr); intrinsic);

            let ptr = ptr.load_scalar(fx);
            let val = CPlace::for_ptr(Pointer::new(ptr), ret.layout()).to_cvalue(fx);
            ret.write_cvalue(fx, val);
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.abs.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().iabs(lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.cls.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().cls(lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.rbit.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().bitrev(lane)
            });
        }

        "llvm.aarch64.neon.fcvtns.v4i32.v4f32" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            // Note: Using inline asm instead of fcvt_to_sint as the latter rounds to zero rather than to nearest

            let a_ptr = a.force_stack(fx).0.get_addr(fx);
            let res_place = CPlace::new_stack_slot(fx, ret.layout());
            let res_ptr = res_place.to_ptr().get_addr(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    "ldr     q0, [x0]
                     fcvtns  v0.4s, v0.4s
                     str     q0, [x1]"
                        .into(),
                )],
                &[
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x0,
                        )),
                        value: a_ptr,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x1,
                        )),
                        value: res_ptr,
                    },
                ],
                InlineAsmOptions::NOSTACK,
            );
            let res = res_place.to_cvalue(fx);
            ret.write_cvalue_transmute(fx, res);
        }

        "llvm.aarch64.neon.frecpe.v4f32" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            let a_ptr = a.force_stack(fx).0.get_addr(fx);
            let res_place = CPlace::new_stack_slot(fx, ret.layout());
            let res_ptr = res_place.to_ptr().get_addr(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    "ldr     q0, [x0]
                     frecpe  v0.4s, v0.4s
                     str     q0, [x1]"
                        .into(),
                )],
                &[
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x0,
                        )),
                        value: a_ptr,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x1,
                        )),
                        value: res_ptr,
                    },
                ],
                InlineAsmOptions::NOSTACK,
            );
            let res = res_place.to_cvalue(fx);
            ret.write_cvalue_transmute(fx, res);
        }

        "llvm.aarch64.neon.frecps.v4f32" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a_ptr = a.force_stack(fx).0.get_addr(fx);
            let b_ptr = b.force_stack(fx).0.get_addr(fx);
            let res_place = CPlace::new_stack_slot(fx, ret.layout());
            let res_ptr = res_place.to_ptr().get_addr(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    "ldr     q0, [x0]
                     ldr     q1, [x1]
                     frecps  v0.4s, v0.4s, v1.4s
                     str     q0, [x2]"
                        .into(),
                )],
                &[
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x0,
                        )),
                        value: a_ptr,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x1,
                        )),
                        value: b_ptr,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x2,
                        )),
                        value: res_ptr,
                    },
                ],
                InlineAsmOptions::NOSTACK,
            );
            let res = res_place.to_cvalue(fx);
            ret.write_cvalue_transmute(fx, res);
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqadd.v")
            || intrinsic.starts_with("llvm.aarch64.neon.uqadd.v") =>
        {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane_typed(fx, x, y, ret, &|fx, x_lane, y_lane| {
                crate::num::codegen_saturating_int_binop(fx, BinOp::Add, x_lane, y_lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqsub.v")
            || intrinsic.starts_with("llvm.aarch64.neon.uqsub.v") =>
        {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane_typed(fx, x, y, ret, &|fx, x_lane, y_lane| {
                crate::num::codegen_saturating_int_binop(fx, BinOp::Sub, x_lane, y_lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smax.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umax.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.fmax.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().fmax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.fmin.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().fmin(x_lane, y_lane),
            );
        }

        "llvm.aarch64.neon.uaddlv.i32.v16i8" => {
            intrinsic_args!(fx, args => (v); intrinsic);

            let mut res_val = fx.bcx.ins().iconst(types::I16, 0);
            for lane_idx in 0..16 {
                let lane = v.value_lane(fx, lane_idx).load_scalar(fx);
                let lane = fx.bcx.ins().uextend(types::I16, lane);
                res_val = fx.bcx.ins().iadd(res_val, lane);
            }
            let res = CValue::by_val(
                fx.bcx.ins().uextend(types::I32, res_val),
                fx.layout_of(fx.tcx.types.i32),
            );
            ret.write_cvalue(fx, res);
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.faddv.f32.v") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| fx.bcx.ins().fadd(a, b));
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smaxv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umaxv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smin.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::SignedLessThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umin.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sminv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::SignedLessThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.uminv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umaxp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().umax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smaxp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().smax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.uminp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().umin(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sminp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().smin(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.fminp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().fmin(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.fmaxp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().fmax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.addp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().iadd(x_lane, y_lane),
            );
        }

        // FIXME generalize vector types
        "llvm.aarch64.neon.tbl1.v8i8" => {
            intrinsic_args!(fx, args => (t, idx); intrinsic);

            let zero = fx.bcx.ins().iconst(types::I8, 0);
            for i in 0..8 {
                let idx_lane = idx.value_lane(fx, i).load_scalar(fx);
                let is_zero =
                    fx.bcx.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, idx_lane, 16);
                let t_idx = fx.bcx.ins().uextend(fx.pointer_type, idx_lane);
                let t_lane = t.value_lane_dyn(fx, t_idx).load_scalar(fx);
                let res = fx.bcx.ins().select(is_zero, zero, t_lane);
                ret.place_lane(fx, i).to_ptr().store(fx, res, MemFlags::trusted());
            }
        }
        "llvm.aarch64.neon.tbl1.v16i8" => {
            intrinsic_args!(fx, args => (t, idx); intrinsic);

            let zero = fx.bcx.ins().iconst(types::I8, 0);
            for i in 0..16 {
                let idx_lane = idx.value_lane(fx, i).load_scalar(fx);
                let is_zero =
                    fx.bcx.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, idx_lane, 16);
                let t_idx = fx.bcx.ins().uextend(fx.pointer_type, idx_lane);
                let t_lane = t.value_lane_dyn(fx, t_idx).load_scalar(fx);
                let res = fx.bcx.ins().select(is_zero, zero, t_lane);
                ret.place_lane(fx, i).to_ptr().store(fx, res, MemFlags::trusted());
            }
        }

        /*
        _ if intrinsic.starts_with("llvm.aarch64.neon.sshl.v")
            || intrinsic.starts_with("llvm.aarch64.neon.sqshl.v")
            // FIXME split this one out once saturating is implemented
            || intrinsic.starts_with("llvm.aarch64.neon.sqshlu.v") =>
        {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            simd_pair_for_each_lane(fx, a, b, ret, &|fx, _lane_ty, _res_lane_ty, a, b| {
                // FIXME saturate?
                fx.bcx.ins().ishl(a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqshrn.v") => {
            let (a, imm32) = match args {
                [a, imm32] => (a, imm32),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm32 = crate::constant::mir_operand_get_const_val(fx, imm32)
                .expect("llvm.aarch64.neon.sqshrn.v* imm32 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm32
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm32 not scalar: {:?}", imm32))
            {
                imm32 if imm32 < 32 => fx.bcx.ins().sshr_imm(lane, i64::from(imm32 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqshrun.v") => {
            let (a, imm32) = match args {
                [a, imm32] => (a, imm32),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm32 = crate::constant::mir_operand_get_const_val(fx, imm32)
                .expect("llvm.aarch64.neon.sqshrn.v* imm32 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm32
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm32 not scalar: {:?}", imm32))
            {
                imm32 if imm32 < 32 => fx.bcx.ins().ushr_imm(lane, i64::from(imm32 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        */
        "llvm.aarch64.crc32b"
        | "llvm.aarch64.crc32h"
        | "llvm.aarch64.crc32w"
        | "llvm.aarch64.crc32x"
        | "llvm.aarch64.crc32cb"
        | "llvm.aarch64.crc32ch"
        | "llvm.aarch64.crc32cw"
        | "llvm.aarch64.crc32cx" => {
            // ARM ARM v8-A: CRC32{,C}{B,H,W,X}.
            // Backs core::arch::aarch64::__crc32{,c}{b,h,w,d}.
            intrinsic_args!(fx, args => (crc, v); intrinsic);

            let crc = crc.load_scalar(fx);
            let v = v.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.aarch64.crc32b" => "crc32b  w0, w0, w1",
                "llvm.aarch64.crc32h" => "crc32h  w0, w0, w1",
                "llvm.aarch64.crc32w" => "crc32w  w0, w0, w1",
                "llvm.aarch64.crc32x" => "crc32x  w0, w0, x1",
                "llvm.aarch64.crc32cb" => "crc32cb w0, w0, w1",
                "llvm.aarch64.crc32ch" => "crc32ch w0, w0, w1",
                "llvm.aarch64.crc32cw" => "crc32cw w0, w0, w1",
                "llvm.aarch64.crc32cx" => "crc32cx w0, w0, x1",
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x0,
                        )),
                        _late: true,
                        in_value: crc,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x1,
                        )),
                        value: v,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.aese" | "llvm.aarch64.crypto.aesd" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.aarch64.crypto.aese" => "aese v0.16b, v1.16b",
                "llvm.aarch64.crypto.aesd" => "aesd v0.16b, v1.16b",
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.aesmc" | "llvm.aarch64.crypto.aesimc" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            let a = a.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.aarch64.crypto.aesmc" => "aesmc v0.16b, v0.16b",
                "llvm.aarch64.crypto.aesimc" => "aesimc v0.16b, v0.16b",
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[CInlineAsmOperand::InOut {
                    reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                        AArch64InlineAsmReg::v0,
                    )),
                    _late: true,
                    in_value: a,
                    out_place: Some(ret),
                }],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha1c" | "llvm.aarch64.crypto.sha1m" | "llvm.aarch64.crypto.sha1p" => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let c = c.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.aarch64.crypto.sha1c" => {
                    "fmov    s2, w1
                     sha1c   q0, s2, v1.4s"
                }
                "llvm.aarch64.crypto.sha1m" => {
                    "fmov    s2, w1
                     sha1m   q0, s2, v1.4s"
                }
                "llvm.aarch64.crypto.sha1p" => {
                    "fmov    s2, w1
                     sha1p   q0, s2, v1.4s"
                }
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: c,
                    },
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v2,
                        )),
                        late: true,
                        place: None,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha1h" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            let a = a.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    "fmov    s0, w0
                     sha1h   s0, s0
                     fmov    w0, s0"
                        .into(),
                )],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        late: true,
                        place: None,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha1su0" => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let c = c.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha1su0 v0.4s, v1.4s, v2.4s".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v2,
                        )),
                        value: c,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha1su1" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha1su1 v0.4s, v1.4s".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha256h" | "llvm.aarch64.crypto.sha256h2" => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let c = c.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.aarch64.crypto.sha256h" => "sha256h q0, q1, v2.4s",
                "llvm.aarch64.crypto.sha256h2" => "sha256h2 q0, q1, v2.4s",
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v2,
                        )),
                        value: c,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha256su0" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha256su0 v0.4s, v1.4s".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha256su1" => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let c = c.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha256su1 v0.4s, v1.4s, v2.4s".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v2,
                        )),
                        value: c,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha512h" | "llvm.aarch64.crypto.sha512h2" => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let c = c.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.aarch64.crypto.sha512h" => "sha512h q0, q1, v2.2d",
                "llvm.aarch64.crypto.sha512h2" => "sha512h2 q0, q1, v2.2d",
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v2,
                        )),
                        value: c,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha512su0" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha512su0 v0.2d, v1.2d".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.sha512su1" => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let c = c.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha512su1 v0.2d, v1.2d, v2.2d".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v2,
                        )),
                        value: c,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.crypto.eor3s.v2i64"
        | "llvm.aarch64.crypto.eor3s.v4i32"
        | "llvm.aarch64.crypto.eor3s.v8i16"
        | "llvm.aarch64.crypto.eor3s.v16i8"
        | "llvm.aarch64.crypto.eor3u.v2i64"
        | "llvm.aarch64.crypto.eor3u.v4i32"
        | "llvm.aarch64.crypto.eor3u.v8i16"
        | "llvm.aarch64.crypto.eor3u.v16i8" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/EOR3--Three-way-exclusive-OR-
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            simd_trio_for_each_lane(
                fx,
                a,
                b,
                c,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane, c_lane| {
                    let xor = fx.bcx.ins().bxor(a_lane, b_lane);
                    fx.bcx.ins().bxor(xor, c_lane)
                },
            );
        }

        "llvm.aarch64.crypto.bcaxs.v2i64"
        | "llvm.aarch64.crypto.bcaxs.v4i32"
        | "llvm.aarch64.crypto.bcaxs.v8i16"
        | "llvm.aarch64.crypto.bcaxs.v16i8"
        | "llvm.aarch64.crypto.bcaxu.v2i64"
        | "llvm.aarch64.crypto.bcaxu.v4i32"
        | "llvm.aarch64.crypto.bcaxu.v8i16"
        | "llvm.aarch64.crypto.bcaxu.v16i8" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/BCAX--Bit-clear-and-exclusive-OR-
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            simd_trio_for_each_lane(
                fx,
                a,
                b,
                c,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane, c_lane| {
                    let band_not = fx.bcx.ins().band_not(b_lane, c_lane);
                    fx.bcx.ins().bxor(a_lane, band_not)
                },
            );
        }

        "llvm.aarch64.crypto.rax1" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/RAX1--Rotate-and-exclusive-OR-
            intrinsic_args!(fx, args => (a, b); intrinsic);

            simd_pair_for_each_lane(
                fx,
                a,
                b,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane| {
                    let rot = fx.bcx.ins().rotl_imm(b_lane, 1);
                    fx.bcx.ins().bxor(a_lane, rot)
                },
            );
        }

        "llvm.aarch64.crypto.xar" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/XAR--Exclusive-OR-and-rotate-
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            let c = c.load_scalar(fx);

            simd_pair_for_each_lane(
                fx,
                a,
                b,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane| {
                    let xor = fx.bcx.ins().bxor(a_lane, b_lane);
                    fx.bcx.ins().rotr(xor, c)
                },
            );
        }

        "llvm.aarch64.neon.pmull64" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    "fmov    d0, x0
                     fmov    d1, x1
                     pmull   v0.1q, v0.1d, v1.1d"
                        .into(),
                )],
                &[
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        late: true,
                        place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x0,
                        )),
                        value: a,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::x1,
                        )),
                        value: b,
                    },
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        late: true,
                        place: None,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.neon.pmull.v8i16" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("pmull v0.8h, v0.8b, v1.8b".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v0,
                        )),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::AArch64(
                            AArch64InlineAsmReg::v1,
                        )),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.aarch64.neon.sqdmulh.v2i32"
        | "llvm.aarch64.neon.sqdmulh.v4i16"
        | "llvm.aarch64.neon.sqdmulh.v4i32"
        | "llvm.aarch64.neon.sqdmulh.v8i16" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/SQDMULH--vector---Signed-saturating-doubling-multiply-returning-high-half-
            intrinsic_args!(fx, args => (a, b); intrinsic);

            // Simplify the "double and shift by esize" into "shift by esize - 1".
            // https://github.com/qemu/qemu/blob/81cc5f39aa3042e9c0b2ea772b42a2c8b1488e76/target/arm/tcg/mve_helper.c#L1267-L1283
            let (result_ty, product_ty, shift, max) = match intrinsic {
                "llvm.aarch64.neon.sqdmulh.v4i16" | "llvm.aarch64.neon.sqdmulh.v8i16" => {
                    (types::I16, types::I32, 15, i64::from(i16::MAX))
                }
                "llvm.aarch64.neon.sqdmulh.v2i32" | "llvm.aarch64.neon.sqdmulh.v4i32" => {
                    (types::I32, types::I64, 31, i64::from(i32::MAX))
                }
                _ => unreachable!(),
            };

            simd_pair_for_each_lane(
                fx,
                a,
                b,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane| {
                    let a_lane = fx.bcx.ins().sextend(product_ty, a_lane);
                    let b_lane = fx.bcx.ins().sextend(product_ty, b_lane);
                    let product = fx.bcx.ins().imul(a_lane, b_lane);
                    let product = fx.bcx.ins().sshr_imm(product, shift);
                    let max = fx.bcx.ins().iconst(product_ty, max);
                    let result = fx.bcx.ins().smin(product, max);
                    fx.bcx.ins().ireduce(result_ty, result)
                },
            );
        }

        "llvm.aarch64.neon.saddlp.v1i64.v2i32"
        | "llvm.aarch64.neon.saddlp.v2i32.v4i16"
        | "llvm.aarch64.neon.saddlp.v2i64.v4i32"
        | "llvm.aarch64.neon.saddlp.v4i16.v8i8"
        | "llvm.aarch64.neon.saddlp.v4i32.v8i16"
        | "llvm.aarch64.neon.saddlp.v8i16.v16i8" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/SADDLP--Signed-add-long-pairwise-
            intrinsic_args!(fx, args => (a); intrinsic);

            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);
            let wide_ty = fx.clif_type(ret_lane_ty).unwrap();

            for lane_idx in 0..ret_lane_count {
                let base = lane_idx * 2;
                let a_lane0 = a.value_lane(fx, base).load_scalar(fx);
                let a_lane1 = a.value_lane(fx, base + 1).load_scalar(fx);
                let a_lane0 = fx.bcx.ins().sextend(wide_ty, a_lane0);
                let a_lane1 = fx.bcx.ins().sextend(wide_ty, a_lane1);
                let sum = fx.bcx.ins().iadd(a_lane0, a_lane1);
                let res_lane = CValue::by_val(sum, ret_lane_layout);
                ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
            }
        }

        "llvm.aarch64.neon.uaddlp.v1i64.v2i32"
        | "llvm.aarch64.neon.uaddlp.v2i32.v4i16"
        | "llvm.aarch64.neon.uaddlp.v2i64.v4i32"
        | "llvm.aarch64.neon.uaddlp.v4i16.v8i8"
        | "llvm.aarch64.neon.uaddlp.v4i32.v8i16"
        | "llvm.aarch64.neon.uaddlp.v8i16.v16i8" => {
            // https://developer.arm.com/documentation/ddi0602/2026-03/SIMD-FP-Instructions/UADDLP--Unsigned-add-long-pairwise-
            intrinsic_args!(fx, args => (a); intrinsic);

            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);
            let wide_ty = fx.clif_type(ret_lane_ty).unwrap();

            for lane_idx in 0..ret_lane_count {
                let base = lane_idx * 2;
                let a_lane0 = a.value_lane(fx, base).load_scalar(fx);
                let a_lane1 = a.value_lane(fx, base + 1).load_scalar(fx);
                let a_lane0 = fx.bcx.ins().uextend(wide_ty, a_lane0);
                let a_lane1 = fx.bcx.ins().uextend(wide_ty, a_lane1);
                let sum = fx.bcx.ins().iadd(a_lane0, a_lane1);
                let res_lane = CValue::by_val(sum, ret_lane_layout);
                ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
            }
        }

        _ => {
            fx.tcx.dcx().warn(format!(
                "unsupported AArch64 llvm intrinsic {}; replacing with trap",
                intrinsic
            ));
            let msg = format!(
                "{intrinsic} is not yet supported.\n\
                 See https://github.com/rust-lang/rustc_codegen_cranelift/issues/171\n\
                 Please open an issue at https://github.com/rust-lang/rustc_codegen_cranelift/issues"
            );
            crate::base::codegen_panic_nounwind(fx, &msg, fx.mir.span);
            return;
        }
    }

    let dest = target.expect("all llvm intrinsics used by stdlib should return");
    let ret_block = fx.get_block(dest);
    fx.bcx.ins().jump(ret_block, &[]);
}
