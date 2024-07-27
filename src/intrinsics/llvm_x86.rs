//! Emulate x86 LLVM intrinsics

use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_target::asm::*;

use crate::inline_asm::{codegen_inline_asm_inner, CInlineAsmOperand};
use crate::intrinsics::*;
use crate::prelude::*;

pub(crate) fn codegen_x86_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    args: &[Spanned<mir::Operand<'tcx>>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
    span: Span,
) {
    match intrinsic {
        "llvm.x86.sse2.pause" | "llvm.aarch64.isb" => {
            // Spin loop hint
        }

        "llvm.x86.avx.vzeroupper" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_zeroupper&ig_expand=7218
            // Do nothing. It is a perf hint anyway.
        }

        // Used by is_x86_feature_detected!();
        "llvm.x86.xgetbv" => {
            intrinsic_args!(fx, args => (xcr_no); intrinsic);

            let xcr_no = xcr_no.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    "
                    xgetbv
                    // out = rdx << 32 | rax
                    shl rdx, 32
                    or rax, rdx
                    "
                    .into(),
                )],
                &[
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::cx)),
                        value: xcr_no,
                    },
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::ax)),
                        late: true,
                        place: Some(ret),
                    },
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::dx)),
                        late: true,
                        place: None,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sse3.ldu.dq" | "llvm.x86.avx.ldu.dq.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_lddqu_si128&ig_expand=4009
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_lddqu_si256&ig_expand=4010
            intrinsic_args!(fx, args => (ptr); intrinsic);

            // FIXME correctly handle unalignedness
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), ret.layout());
            ret.write_cvalue(fx, val);
        }

        "llvm.x86.avx2.gather.d.d"
        | "llvm.x86.avx2.gather.d.q"
        | "llvm.x86.avx2.gather.d.ps"
        | "llvm.x86.avx2.gather.d.pd"
        | "llvm.x86.avx2.gather.d.d.256"
        | "llvm.x86.avx2.gather.d.q.256"
        | "llvm.x86.avx2.gather.d.ps.256"
        | "llvm.x86.avx2.gather.d.pd.256"
        | "llvm.x86.avx2.gather.q.d"
        | "llvm.x86.avx2.gather.q.q"
        | "llvm.x86.avx2.gather.q.ps"
        | "llvm.x86.avx2.gather.q.pd"
        | "llvm.x86.avx2.gather.q.d.256"
        | "llvm.x86.avx2.gather.q.q.256"
        | "llvm.x86.avx2.gather.q.ps.256"
        | "llvm.x86.avx2.gather.q.pd.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_i64gather_pd&ig_expand=3818
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_i64gather_pd&ig_expand=3819
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_i64gather_pd&ig_expand=3821
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_i64gather_pd&ig_expand=3822
            // ...

            intrinsic_args!(fx, args => (src, ptr, index, mask, scale); intrinsic);

            let (src_lane_count, src_lane_ty) = src.layout().ty.simd_size_and_type(fx.tcx);
            let (index_lane_count, index_lane_ty) = index.layout().ty.simd_size_and_type(fx.tcx);
            let (mask_lane_count, mask_lane_ty) = mask.layout().ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(src_lane_ty, ret_lane_ty);
            assert!(index_lane_ty.is_integral());
            assert_eq!(src_lane_count, mask_lane_count);
            assert_eq!(src_lane_count, ret_lane_count);

            let lane_clif_ty = fx.clif_type(ret_lane_ty).unwrap();
            let index_lane_clif_ty = fx.clif_type(index_lane_ty).unwrap();
            let mask_lane_clif_ty = fx.clif_type(mask_lane_ty).unwrap();
            let ret_lane_layout = fx.layout_of(ret_lane_ty);

            let ptr = ptr.load_scalar(fx);
            let scale = scale.load_scalar(fx);
            let scale = fx.bcx.ins().uextend(types::I64, scale);
            for lane_idx in 0..std::cmp::min(src_lane_count, index_lane_count) {
                let src_lane = src.value_lane(fx, lane_idx).load_scalar(fx);
                let index_lane = index.value_lane(fx, lane_idx).load_scalar(fx);
                let mask_lane = mask.value_lane(fx, lane_idx).load_scalar(fx);
                let mask_lane =
                    fx.bcx.ins().bitcast(mask_lane_clif_ty.as_int(), MemFlags::new(), mask_lane);

                let if_enabled = fx.bcx.create_block();
                let if_disabled = fx.bcx.create_block();
                let next = fx.bcx.create_block();
                let res_lane = fx.bcx.append_block_param(next, lane_clif_ty);

                let mask_lane = match mask_lane_clif_ty {
                    types::I32 | types::F32 => {
                        fx.bcx.ins().band_imm(mask_lane, 0x8000_0000u64 as i64)
                    }
                    types::I64 | types::F64 => {
                        fx.bcx.ins().band_imm(mask_lane, 0x8000_0000_0000_0000u64 as i64)
                    }
                    _ => unreachable!(),
                };
                fx.bcx.ins().brif(mask_lane, if_enabled, &[], if_disabled, &[]);
                fx.bcx.seal_block(if_enabled);
                fx.bcx.seal_block(if_disabled);

                fx.bcx.switch_to_block(if_enabled);
                let index_lane = if index_lane_clif_ty != types::I64 {
                    fx.bcx.ins().sextend(types::I64, index_lane)
                } else {
                    index_lane
                };
                let offset = fx.bcx.ins().imul(index_lane, scale);
                let lane_ptr = fx.bcx.ins().iadd(ptr, offset);
                let res = fx.bcx.ins().load(lane_clif_ty, MemFlags::trusted(), lane_ptr, 0);
                fx.bcx.ins().jump(next, &[res]);

                fx.bcx.switch_to_block(if_disabled);
                fx.bcx.ins().jump(next, &[src_lane]);

                fx.bcx.seal_block(next);
                fx.bcx.switch_to_block(next);

                fx.bcx.ins().nop();

                ret.place_lane(fx, lane_idx)
                    .write_cvalue(fx, CValue::by_val(res_lane, ret_lane_layout));
            }

            for lane_idx in std::cmp::min(src_lane_count, index_lane_count)..ret_lane_count {
                let zero_lane = fx.bcx.ins().iconst(mask_lane_clif_ty.as_int(), 0);
                let zero_lane = fx.bcx.ins().bitcast(mask_lane_clif_ty, MemFlags::new(), zero_lane);
                ret.place_lane(fx, lane_idx)
                    .write_cvalue(fx, CValue::by_val(zero_lane, ret_lane_layout));
            }
        }

        "llvm.x86.sse.add.ss" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_add_ss&ig_expand=171
            intrinsic_args!(fx, args => (a, b); intrinsic);

            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.layout(), ret.layout());
            let layout = a.layout();

            let (_, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            assert!(lane_ty.is_floating_point());
            let ret_lane_layout = fx.layout_of(lane_ty);

            ret.write_cvalue(fx, a);

            let a_lane = a.value_lane(fx, 0).load_scalar(fx);
            let b_lane = b.value_lane(fx, 0).load_scalar(fx);

            let res = fx.bcx.ins().fadd(a_lane, b_lane);

            let res_lane = CValue::by_val(res, ret_lane_layout);
            ret.place_lane(fx, 0).write_cvalue(fx, res_lane);
        }

        "llvm.x86.sse.sqrt.ps" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sqrt_ps&ig_expand=6245
            intrinsic_args!(fx, args => (a); intrinsic);

            // FIXME use vector instructions when possible
            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().sqrt(lane)
            });
        }

        "llvm.x86.sse.max.ps" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_max_ps&ig_expand=4357
            intrinsic_args!(fx, args => (a, b); intrinsic);

            simd_pair_for_each_lane(
                fx,
                a,
                b,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane| fx.bcx.ins().fmax(a_lane, b_lane),
            );
        }

        "llvm.x86.sse.min.ps" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_min_ps&ig_expand=4489
            intrinsic_args!(fx, args => (a, b); intrinsic);

            simd_pair_for_each_lane(
                fx,
                a,
                b,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane| fx.bcx.ins().fmin(a_lane, b_lane),
            );
        }

        "llvm.x86.sse.cmp.ps" | "llvm.x86.sse2.cmp.pd" => {
            let (x, y, kind) = match args {
                [x, y, kind] => (x, y, kind),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let x = codegen_operand(fx, &x.node);
            let y = codegen_operand(fx, &y.node);
            let kind = match &kind.node {
                Operand::Constant(const_) => crate::constant::eval_mir_constant(fx, const_).0,
                Operand::Copy(_) | Operand::Move(_) => unreachable!("{kind:?}"),
            };

            let flt_cc = match kind
                .try_to_bits(Size::from_bytes(1))
                .unwrap_or_else(|| panic!("kind not scalar: {:?}", kind))
                .try_into()
                .unwrap()
            {
                _CMP_EQ_OQ | _CMP_EQ_OS => FloatCC::Equal,
                _CMP_LT_OS | _CMP_LT_OQ => FloatCC::LessThan,
                _CMP_LE_OS | _CMP_LE_OQ => FloatCC::LessThanOrEqual,
                _CMP_UNORD_Q | _CMP_UNORD_S => FloatCC::Unordered,
                _CMP_NEQ_UQ | _CMP_NEQ_US => FloatCC::NotEqual,
                _CMP_NLT_US | _CMP_NLT_UQ => FloatCC::UnorderedOrGreaterThanOrEqual,
                _CMP_NLE_US | _CMP_NLE_UQ => FloatCC::UnorderedOrGreaterThan,
                _CMP_ORD_Q | _CMP_ORD_S => FloatCC::Ordered,
                _CMP_EQ_UQ | _CMP_EQ_US => FloatCC::UnorderedOrEqual,
                _CMP_NGE_US | _CMP_NGE_UQ => FloatCC::UnorderedOrLessThan,
                _CMP_NGT_US | _CMP_NGT_UQ => FloatCC::UnorderedOrLessThanOrEqual,
                _CMP_FALSE_OQ | _CMP_FALSE_OS => todo!(),
                _CMP_NEQ_OQ | _CMP_NEQ_OS => FloatCC::OrderedNotEqual,
                _CMP_GE_OS | _CMP_GE_OQ => FloatCC::GreaterThanOrEqual,
                _CMP_GT_OS | _CMP_GT_OQ => FloatCC::GreaterThan,
                _CMP_TRUE_UQ | _CMP_TRUE_US => todo!(),

                kind => unreachable!("kind {:?}", kind),
            };

            // Copied from stdarch
            /// Equal (ordered, non-signaling)
            const _CMP_EQ_OQ: i32 = 0x00;
            /// Less-than (ordered, signaling)
            const _CMP_LT_OS: i32 = 0x01;
            /// Less-than-or-equal (ordered, signaling)
            const _CMP_LE_OS: i32 = 0x02;
            /// Unordered (non-signaling)
            const _CMP_UNORD_Q: i32 = 0x03;
            /// Not-equal (unordered, non-signaling)
            const _CMP_NEQ_UQ: i32 = 0x04;
            /// Not-less-than (unordered, signaling)
            const _CMP_NLT_US: i32 = 0x05;
            /// Not-less-than-or-equal (unordered, signaling)
            const _CMP_NLE_US: i32 = 0x06;
            /// Ordered (non-signaling)
            const _CMP_ORD_Q: i32 = 0x07;
            /// Equal (unordered, non-signaling)
            const _CMP_EQ_UQ: i32 = 0x08;
            /// Not-greater-than-or-equal (unordered, signaling)
            const _CMP_NGE_US: i32 = 0x09;
            /// Not-greater-than (unordered, signaling)
            const _CMP_NGT_US: i32 = 0x0a;
            /// False (ordered, non-signaling)
            const _CMP_FALSE_OQ: i32 = 0x0b;
            /// Not-equal (ordered, non-signaling)
            const _CMP_NEQ_OQ: i32 = 0x0c;
            /// Greater-than-or-equal (ordered, signaling)
            const _CMP_GE_OS: i32 = 0x0d;
            /// Greater-than (ordered, signaling)
            const _CMP_GT_OS: i32 = 0x0e;
            /// True (unordered, non-signaling)
            const _CMP_TRUE_UQ: i32 = 0x0f;
            /// Equal (ordered, signaling)
            const _CMP_EQ_OS: i32 = 0x10;
            /// Less-than (ordered, non-signaling)
            const _CMP_LT_OQ: i32 = 0x11;
            /// Less-than-or-equal (ordered, non-signaling)
            const _CMP_LE_OQ: i32 = 0x12;
            /// Unordered (signaling)
            const _CMP_UNORD_S: i32 = 0x13;
            /// Not-equal (unordered, signaling)
            const _CMP_NEQ_US: i32 = 0x14;
            /// Not-less-than (unordered, non-signaling)
            const _CMP_NLT_UQ: i32 = 0x15;
            /// Not-less-than-or-equal (unordered, non-signaling)
            const _CMP_NLE_UQ: i32 = 0x16;
            /// Ordered (signaling)
            const _CMP_ORD_S: i32 = 0x17;
            /// Equal (unordered, signaling)
            const _CMP_EQ_US: i32 = 0x18;
            /// Not-greater-than-or-equal (unordered, non-signaling)
            const _CMP_NGE_UQ: i32 = 0x19;
            /// Not-greater-than (unordered, non-signaling)
            const _CMP_NGT_UQ: i32 = 0x1a;
            /// False (ordered, signaling)
            const _CMP_FALSE_OS: i32 = 0x1b;
            /// Not-equal (ordered, signaling)
            const _CMP_NEQ_OS: i32 = 0x1c;
            /// Greater-than-or-equal (ordered, non-signaling)
            const _CMP_GE_OQ: i32 = 0x1d;
            /// Greater-than (ordered, non-signaling)
            const _CMP_GT_OQ: i32 = 0x1e;
            /// True (unordered, signaling)
            const _CMP_TRUE_US: i32 = 0x1f;

            simd_pair_for_each_lane(fx, x, y, ret, &|fx, lane_ty, res_lane_ty, x_lane, y_lane| {
                let res_lane = match lane_ty.kind() {
                    ty::Float(_) => fx.bcx.ins().fcmp(flt_cc, x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_ty),
                };
                bool_to_zero_or_max_uint(fx, res_lane_ty, res_lane)
            });
        }
        "llvm.x86.ssse3.pshuf.b.128" | "llvm.x86.avx2.pshuf.b" => {
            let (a, b) = match args {
                [a, b] => (a, b),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, &a.node);
            let b = codegen_operand(fx, &b.node);

            // Based on the pseudocode at https://github.com/rust-lang/stdarch/blob/1cfbca8b38fd9b4282b2f054f61c6ca69fc7ce29/crates/core_arch/src/x86/avx2.rs#L2319-L2332
            let zero = fx.bcx.ins().iconst(types::I8, 0);
            for i in 0..16 {
                let b_lane = b.value_lane(fx, i).load_scalar(fx);
                let is_zero = fx.bcx.ins().band_imm(b_lane, 0x80);
                let a_idx = fx.bcx.ins().band_imm(b_lane, 0xf);
                let a_idx = fx.bcx.ins().uextend(fx.pointer_type, a_idx);
                let a_lane = a.value_lane_dyn(fx, a_idx).load_scalar(fx);
                let res = fx.bcx.ins().select(is_zero, zero, a_lane);
                ret.place_lane(fx, i).to_ptr().store(fx, res, MemFlags::trusted());
            }

            if intrinsic == "llvm.x86.avx2.pshuf.b" {
                for i in 16..32 {
                    let b_lane = b.value_lane(fx, i).load_scalar(fx);
                    let is_zero = fx.bcx.ins().band_imm(b_lane, 0x80);
                    let b_lane_masked = fx.bcx.ins().band_imm(b_lane, 0xf);
                    let a_idx = fx.bcx.ins().iadd_imm(b_lane_masked, 16);
                    let a_idx = fx.bcx.ins().uextend(fx.pointer_type, a_idx);
                    let a_lane = a.value_lane_dyn(fx, a_idx).load_scalar(fx);
                    let res = fx.bcx.ins().select(is_zero, zero, a_lane);
                    ret.place_lane(fx, i).to_ptr().store(fx, res, MemFlags::trusted());
                }
            }
        }
        "llvm.x86.avx2.permd" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permutevar8x32_epi32
            intrinsic_args!(fx, args => (a, idx); intrinsic);

            for j in 0..=7 {
                let index = idx.value_typed_lane(fx, fx.tcx.types.u32, j).load_scalar(fx);
                let index = fx.bcx.ins().uextend(fx.pointer_type, index);
                let value = a.value_lane_dyn(fx, index).load_scalar(fx);
                ret.place_typed_lane(fx, fx.tcx.types.u32, j).to_ptr().store(
                    fx,
                    value,
                    MemFlags::trusted(),
                );
            }
        }
        "llvm.x86.avx2.vperm2i128"
        | "llvm.x86.avx.vperm2f128.ps.256"
        | "llvm.x86.avx.vperm2f128.pd.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permute2x128_si256
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permute2f128_ps
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permute2f128_pd
            let (a, b, imm8) = match args {
                [a, b, imm8] => (a, b, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, &a.node);
            let b = codegen_operand(fx, &b.node);
            let imm8 = codegen_operand(fx, &imm8.node).load_scalar(fx);

            let a_low = a.value_typed_lane(fx, fx.tcx.types.u128, 0).load_scalar(fx);
            let a_high = a.value_typed_lane(fx, fx.tcx.types.u128, 1).load_scalar(fx);

            let b_low = b.value_typed_lane(fx, fx.tcx.types.u128, 0).load_scalar(fx);
            let b_high = b.value_typed_lane(fx, fx.tcx.types.u128, 1).load_scalar(fx);

            fn select4(
                fx: &mut FunctionCx<'_, '_, '_>,
                a_high: Value,
                a_low: Value,
                b_high: Value,
                b_low: Value,
                control: Value,
            ) -> Value {
                let a_or_b = fx.bcx.ins().band_imm(control, 0b0010);
                let high_or_low = fx.bcx.ins().band_imm(control, 0b0001);
                let is_zero = fx.bcx.ins().band_imm(control, 0b1000);

                let zero = fx.bcx.ins().iconst(types::I64, 0);
                let zero = fx.bcx.ins().iconcat(zero, zero);

                let res_a = fx.bcx.ins().select(high_or_low, a_high, a_low);
                let res_b = fx.bcx.ins().select(high_or_low, b_high, b_low);
                let res = fx.bcx.ins().select(a_or_b, res_b, res_a);
                fx.bcx.ins().select(is_zero, zero, res)
            }

            let control0 = imm8;
            let res_low = select4(fx, a_high, a_low, b_high, b_low, control0);

            let control1 = fx.bcx.ins().ushr_imm(imm8, 4);
            let res_high = select4(fx, a_high, a_low, b_high, b_low, control1);

            ret.place_typed_lane(fx, fx.tcx.types.u128, 0).to_ptr().store(
                fx,
                res_low,
                MemFlags::trusted(),
            );
            ret.place_typed_lane(fx, fx.tcx.types.u128, 1).to_ptr().store(
                fx,
                res_high,
                MemFlags::trusted(),
            );
        }
        "llvm.x86.ssse3.pabs.b.128" | "llvm.x86.ssse3.pabs.w.128" | "llvm.x86.ssse3.pabs.d.128" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().iabs(lane)
            });
        }
        "llvm.x86.sse2.cvttps2dq" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttps_epi32&ig_expand=2429
            intrinsic_args!(fx, args => (a); intrinsic);
            let a = a.load_scalar(fx);

            let value = fx.bcx.ins().x86_cvtt2dq(types::I32X4, a);
            let cvalue = CValue::by_val(value, ret.layout());
            ret.write_cvalue(fx, cvalue);
        }
        "llvm.x86.sse2.cvtps2dq" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtps_epi32
            intrinsic_args!(fx, args => (a); intrinsic);
            let a = a.load_scalar(fx);

            // Using inline asm instead of fcvt_to_sint_sat as unrepresentable values are turned
            // into 0x80000000 for which Cranelift doesn't have a native instruction.
            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("cvtps2dq xmm0, xmm0".into())],
                &[CInlineAsmOperand::InOut {
                    reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                    _late: true,
                    in_value: a,
                    out_place: Some(ret),
                }],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }
        "llvm.x86.addcarry.32" | "llvm.x86.addcarry.64" => {
            intrinsic_args!(fx, args => (c_in, a, b); intrinsic);
            let c_in = c_in.load_scalar(fx);

            let (cb_out, c) = llvm_add_sub(fx, BinOp::Add, c_in, a, b);

            let layout = fx.layout_of(Ty::new_tup(fx.tcx, &[fx.tcx.types.u8, a.layout().ty]));
            let val = CValue::by_val_pair(cb_out, c, layout);
            ret.write_cvalue(fx, val);
        }
        "llvm.x86.addcarryx.u32" | "llvm.x86.addcarryx.u64" => {
            intrinsic_args!(fx, args => (c_in, a, b, out); intrinsic);
            let c_in = c_in.load_scalar(fx);

            let (cb_out, c) = llvm_add_sub(fx, BinOp::Add, c_in, a, b);

            Pointer::new(out.load_scalar(fx)).store(fx, c, MemFlags::trusted());
            ret.write_cvalue(fx, CValue::by_val(cb_out, fx.layout_of(fx.tcx.types.u8)));
        }
        "llvm.x86.subborrow.32" | "llvm.x86.subborrow.64" => {
            intrinsic_args!(fx, args => (b_in, a, b); intrinsic);
            let b_in = b_in.load_scalar(fx);

            let (cb_out, c) = llvm_add_sub(fx, BinOp::Sub, b_in, a, b);

            let layout = fx.layout_of(Ty::new_tup(fx.tcx, &[fx.tcx.types.u8, a.layout().ty]));
            let val = CValue::by_val_pair(cb_out, c, layout);
            ret.write_cvalue(fx, val);
        }
        "llvm.x86.sse2.pavg.b" | "llvm.x86.sse2.pavg.w" => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            // FIXME use vector instructions when possible
            simd_pair_for_each_lane(
                fx,
                a,
                b,
                ret,
                &|fx, _lane_ty, _res_lane_ty, a_lane, b_lane| {
                    // (a + b + 1) >> 1
                    let lane_ty = fx.bcx.func.dfg.value_type(a_lane);
                    let a_lane = fx.bcx.ins().uextend(lane_ty.double_width().unwrap(), a_lane);
                    let b_lane = fx.bcx.ins().uextend(lane_ty.double_width().unwrap(), b_lane);
                    let sum = fx.bcx.ins().iadd(a_lane, b_lane);
                    let num_plus_one = fx.bcx.ins().iadd_imm(sum, 1);
                    let res = fx.bcx.ins().ushr_imm(num_plus_one, 1);
                    fx.bcx.ins().ireduce(lane_ty, res)
                },
            );
        }
        "llvm.x86.sse2.psra.w" => {
            intrinsic_args!(fx, args => (a, count); intrinsic);

            let count_lane = count.force_stack(fx).0.load(fx, types::I64, MemFlags::trusted());
            let lane_ty = fx.clif_type(a.layout().ty.simd_size_and_type(fx.tcx).1).unwrap();
            let max_count = fx.bcx.ins().iconst(types::I64, i64::from(lane_ty.bits() - 1));
            let saturated_count = fx.bcx.ins().umin(count_lane, max_count);

            // FIXME use vector instructions when possible
            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, a_lane| {
                fx.bcx.ins().sshr(a_lane, saturated_count)
            });
        }
        "llvm.x86.sse2.psad.bw" | "llvm.x86.avx2.psad.bw" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sad_epu8&ig_expand=5770
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_sad_epu8&ig_expand=5771
            intrinsic_args!(fx, args => (a, b); intrinsic);

            assert_eq!(a.layout(), b.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(lane_ty, fx.tcx.types.u8);
            assert_eq!(ret_lane_ty, fx.tcx.types.u64);
            assert_eq!(lane_count, ret_lane_count * 8);

            let ret_lane_layout = fx.layout_of(fx.tcx.types.u64);
            for out_lane_idx in 0..lane_count / 8 {
                let mut lane_diff_acc = fx.bcx.ins().iconst(types::I64, 0);

                for lane_idx in out_lane_idx * 8..out_lane_idx * 8 + 8 {
                    let a_lane = a.value_lane(fx, lane_idx).load_scalar(fx);
                    let a_lane = fx.bcx.ins().uextend(types::I16, a_lane);
                    let b_lane = b.value_lane(fx, lane_idx).load_scalar(fx);
                    let b_lane = fx.bcx.ins().uextend(types::I16, b_lane);

                    let lane_diff = fx.bcx.ins().isub(a_lane, b_lane);
                    let abs_lane_diff = fx.bcx.ins().iabs(lane_diff);
                    let abs_lane_diff = fx.bcx.ins().uextend(types::I64, abs_lane_diff);
                    lane_diff_acc = fx.bcx.ins().iadd(lane_diff_acc, abs_lane_diff);
                }

                let res_lane = CValue::by_val(lane_diff_acc, ret_lane_layout);

                ret.place_lane(fx, out_lane_idx).write_cvalue(fx, res_lane);
            }
        }
        "llvm.x86.ssse3.pmadd.ub.sw.128" | "llvm.x86.avx2.pmadd.ub.sw" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maddubs_epi16&ig_expand=4267
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maddubs_epi16&ig_expand=4270
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let (lane_count, lane_ty) = a.layout().ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(lane_ty, fx.tcx.types.u8);
            assert_eq!(ret_lane_ty, fx.tcx.types.i16);
            assert_eq!(lane_count, ret_lane_count * 2);

            let ret_lane_layout = fx.layout_of(fx.tcx.types.i16);
            for out_lane_idx in 0..lane_count / 2 {
                let a_lane0 = a.value_lane(fx, out_lane_idx * 2).load_scalar(fx);
                let a_lane0 = fx.bcx.ins().uextend(types::I16, a_lane0);
                let b_lane0 = b.value_lane(fx, out_lane_idx * 2).load_scalar(fx);
                let b_lane0 = fx.bcx.ins().sextend(types::I16, b_lane0);

                let a_lane1 = a.value_lane(fx, out_lane_idx * 2 + 1).load_scalar(fx);
                let a_lane1 = fx.bcx.ins().uextend(types::I16, a_lane1);
                let b_lane1 = b.value_lane(fx, out_lane_idx * 2 + 1).load_scalar(fx);
                let b_lane1 = fx.bcx.ins().sextend(types::I16, b_lane1);

                let mul0: Value = fx.bcx.ins().imul(a_lane0, b_lane0);
                let mul1 = fx.bcx.ins().imul(a_lane1, b_lane1);

                let (val, has_overflow) = fx.bcx.ins().sadd_overflow(mul0, mul1);

                let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, mul1, 0);

                let min = fx.bcx.ins().iconst(types::I16, i64::from(i16::MIN as u16));
                let max = fx.bcx.ins().iconst(types::I16, i64::from(i16::MAX as u16));

                let sat_val = fx.bcx.ins().select(rhs_ge_zero, max, min);
                let res_lane = fx.bcx.ins().select(has_overflow, sat_val, val);

                let res_lane = CValue::by_val(res_lane, ret_lane_layout);

                ret.place_lane(fx, out_lane_idx).write_cvalue(fx, res_lane);
            }
        }
        "llvm.x86.sse2.pmadd.wd" | "llvm.x86.avx2.pmadd.wd" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_madd_epi16&ig_expand=4231
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_madd_epi16&ig_expand=4234
            intrinsic_args!(fx, args => (a, b); intrinsic);

            assert_eq!(a.layout(), b.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(lane_ty, fx.tcx.types.i16);
            assert_eq!(ret_lane_ty, fx.tcx.types.i32);
            assert_eq!(lane_count, ret_lane_count * 2);

            let ret_lane_layout = fx.layout_of(fx.tcx.types.i32);
            for out_lane_idx in 0..lane_count / 2 {
                let a_lane0 = a.value_lane(fx, out_lane_idx * 2).load_scalar(fx);
                let a_lane0 = fx.bcx.ins().sextend(types::I32, a_lane0);
                let b_lane0 = b.value_lane(fx, out_lane_idx * 2).load_scalar(fx);
                let b_lane0 = fx.bcx.ins().sextend(types::I32, b_lane0);

                let a_lane1 = a.value_lane(fx, out_lane_idx * 2 + 1).load_scalar(fx);
                let a_lane1 = fx.bcx.ins().sextend(types::I32, a_lane1);
                let b_lane1 = b.value_lane(fx, out_lane_idx * 2 + 1).load_scalar(fx);
                let b_lane1 = fx.bcx.ins().sextend(types::I32, b_lane1);

                let mul0: Value = fx.bcx.ins().imul(a_lane0, b_lane0);
                let mul1 = fx.bcx.ins().imul(a_lane1, b_lane1);

                let res_lane = fx.bcx.ins().iadd(mul0, mul1);
                let res_lane = CValue::by_val(res_lane, ret_lane_layout);

                ret.place_lane(fx, out_lane_idx).write_cvalue(fx, res_lane);
            }
        }

        "llvm.x86.ssse3.pmul.hr.sw.128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mulhrs_epi16&ig_expand=4782
            intrinsic_args!(fx, args => (a, b); intrinsic);

            assert_eq!(a.layout(), b.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(lane_ty, fx.tcx.types.i16);
            assert_eq!(ret_lane_ty, fx.tcx.types.i16);
            assert_eq!(lane_count, ret_lane_count);

            let ret_lane_layout = fx.layout_of(fx.tcx.types.i16);
            for out_lane_idx in 0..lane_count {
                let a_lane = a.value_lane(fx, out_lane_idx).load_scalar(fx);
                let a_lane = fx.bcx.ins().sextend(types::I32, a_lane);
                let b_lane = b.value_lane(fx, out_lane_idx).load_scalar(fx);
                let b_lane = fx.bcx.ins().sextend(types::I32, b_lane);

                let mul: Value = fx.bcx.ins().imul(a_lane, b_lane);
                let shifted = fx.bcx.ins().ushr_imm(mul, 14);
                let incremented = fx.bcx.ins().iadd_imm(shifted, 1);
                let shifted_again = fx.bcx.ins().ushr_imm(incremented, 1);

                let res_lane = fx.bcx.ins().ireduce(types::I16, shifted_again);
                let res_lane = CValue::by_val(res_lane, ret_lane_layout);

                ret.place_lane(fx, out_lane_idx).write_cvalue(fx, res_lane);
            }
        }

        "llvm.x86.sse2.packuswb.128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi16&ig_expand=4903
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::U8, PackWidth::Sse);
        }

        "llvm.x86.sse2.packsswb.128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packs_epi16&ig_expand=4848
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::S8, PackWidth::Sse);
        }

        "llvm.x86.avx2.packuswb" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_packus_epi16&ig_expand=4906
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::U8, PackWidth::Avx);
        }

        "llvm.x86.avx2.packsswb" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_packs_epi16&ig_expand=4851
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::S8, PackWidth::Avx);
        }

        "llvm.x86.sse41.packusdw" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi32&ig_expand=4912
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::U16, PackWidth::Sse);
        }

        "llvm.x86.sse2.packssdw.128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packs_epi32&ig_expand=4889
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::S16, PackWidth::Sse);
        }

        "llvm.x86.avx2.packusdw" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_packus_epi32&ig_expand=4883
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::U16, PackWidth::Avx);
        }

        "llvm.x86.avx2.packssdw" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_packs_epi32&ig_expand=4892
            intrinsic_args!(fx, args => (a, b); intrinsic);

            pack_instruction(fx, a, b, ret, PackSize::S16, PackWidth::Avx);
        }

        "llvm.x86.fma.vfmaddsub.ps"
        | "llvm.x86.fma.vfmaddsub.pd"
        | "llvm.x86.fma.vfmaddsub.ps.256"
        | "llvm.x86.fma.vfmaddsub.pd.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmaddsub_ps&ig_expand=3205
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmaddsub_pd&ig_expand=3181
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmaddsub_ps&ig_expand=3209
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmaddsub_pd&ig_expand=3185
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.layout(), c.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert!(lane_ty.is_floating_point());
            assert!(ret_lane_ty.is_floating_point());
            assert_eq!(lane_count, ret_lane_count);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);

            for idx in 0..lane_count {
                let a_lane = a.value_lane(fx, idx).load_scalar(fx);
                let b_lane = b.value_lane(fx, idx).load_scalar(fx);
                let c_lane = c.value_lane(fx, idx).load_scalar(fx);

                let mul = fx.bcx.ins().fmul(a_lane, b_lane);
                let res = if idx & 1 == 0 {
                    fx.bcx.ins().fsub(mul, c_lane)
                } else {
                    fx.bcx.ins().fadd(mul, c_lane)
                };

                let res_lane = CValue::by_val(res, ret_lane_layout);
                ret.place_lane(fx, idx).write_cvalue(fx, res_lane);
            }
        }

        "llvm.x86.fma.vfmsubadd.ps"
        | "llvm.x86.fma.vfmsubadd.pd"
        | "llvm.x86.fma.vfmsubadd.ps.256"
        | "llvm.x86.fma.vfmsubadd.pd.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsubadd_ps&ig_expand=3325
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsubadd_pd&ig_expand=3301
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmsubadd_ps&ig_expand=3329
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmsubadd_pd&ig_expand=3305
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.layout(), c.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert!(lane_ty.is_floating_point());
            assert!(ret_lane_ty.is_floating_point());
            assert_eq!(lane_count, ret_lane_count);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);

            for idx in 0..lane_count {
                let a_lane = a.value_lane(fx, idx).load_scalar(fx);
                let b_lane = b.value_lane(fx, idx).load_scalar(fx);
                let c_lane = c.value_lane(fx, idx).load_scalar(fx);

                let mul = fx.bcx.ins().fmul(a_lane, b_lane);
                let res = if idx & 1 == 0 {
                    fx.bcx.ins().fadd(mul, c_lane)
                } else {
                    fx.bcx.ins().fsub(mul, c_lane)
                };

                let res_lane = CValue::by_val(res, ret_lane_layout);
                ret.place_lane(fx, idx).write_cvalue(fx, res_lane);
            }
        }

        "llvm.x86.fma.vfnmadd.ps"
        | "llvm.x86.fma.vfnmadd.pd"
        | "llvm.x86.fma.vfnmadd.ps.256"
        | "llvm.x86.fma.vfnmadd.pd.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmadd_ps&ig_expand=3391
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmadd_pd&ig_expand=3367
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fnmadd_ps&ig_expand=3395
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fnmadd_pd&ig_expand=3371
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.layout(), c.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert!(lane_ty.is_floating_point());
            assert!(ret_lane_ty.is_floating_point());
            assert_eq!(lane_count, ret_lane_count);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);

            for idx in 0..lane_count {
                let a_lane = a.value_lane(fx, idx).load_scalar(fx);
                let b_lane = b.value_lane(fx, idx).load_scalar(fx);
                let c_lane = c.value_lane(fx, idx).load_scalar(fx);

                let mul = fx.bcx.ins().fmul(a_lane, b_lane);
                let neg_mul = fx.bcx.ins().fneg(mul);
                let res = fx.bcx.ins().fadd(neg_mul, c_lane);

                let res_lane = CValue::by_val(res, ret_lane_layout);
                ret.place_lane(fx, idx).write_cvalue(fx, res_lane);
            }
        }

        "llvm.x86.sse42.crc32.32.8"
        | "llvm.x86.sse42.crc32.32.16"
        | "llvm.x86.sse42.crc32.32.32"
        | "llvm.x86.sse42.crc32.64.64" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=1419&text=_mm_crc32_u32
            intrinsic_args!(fx, args => (crc, v); intrinsic);

            let crc = crc.load_scalar(fx);
            let v = v.load_scalar(fx);

            let asm = match intrinsic {
                "llvm.x86.sse42.crc32.32.8" => "crc32 eax, dl",
                "llvm.x86.sse42.crc32.32.16" => "crc32 eax, dx",
                "llvm.x86.sse42.crc32.32.32" => "crc32 eax, edx",
                "llvm.x86.sse42.crc32.64.64" => "crc32 rax, rdx",
                _ => unreachable!(),
            };

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(asm.into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::ax)),
                        _late: true,
                        in_value: crc,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::dx)),
                        value: v,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sse42.pcmpestri128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestri&ig_expand=939
            intrinsic_args!(fx, args => (a, la, b, lb, _imm8); intrinsic);

            let a = a.load_scalar(fx);
            let la = la.load_scalar(fx);
            let b = b.load_scalar(fx);
            let lb = lb.load_scalar(fx);

            let imm8 =
                if let Some(imm8) = crate::constant::mir_operand_get_const_val(fx, &args[4].node) {
                    imm8
                } else {
                    fx.tcx
                        .dcx()
                        .span_fatal(span, "Index argument for `_mm_cmpestri` is not a constant");
                };

            let imm8 = imm8.to_u8();

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(format!("pcmpestri xmm0, xmm1, {imm8}").into())],
                &[
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        value: a,
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: b,
                    },
                    // Implicit argument to the pcmpestri intrinsic
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::ax)),
                        value: la,
                    },
                    // Implicit argument to the pcmpestri intrinsic
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::dx)),
                        value: lb,
                    },
                    // Implicit result of the pcmpestri intrinsic
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::cx)),
                        late: true,
                        place: Some(ret),
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sse42.pcmpestrm128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestrm&ig_expand=940
            intrinsic_args!(fx, args => (a, la, b, lb, _imm8); intrinsic);

            let a = a.load_scalar(fx);
            let la = la.load_scalar(fx);
            let b = b.load_scalar(fx);
            let lb = lb.load_scalar(fx);

            let imm8 =
                if let Some(imm8) = crate::constant::mir_operand_get_const_val(fx, &args[4].node) {
                    imm8
                } else {
                    fx.tcx
                        .dcx()
                        .span_fatal(span, "Index argument for `_mm_cmpestrm` is not a constant");
                };

            let imm8 = imm8.to_u8();

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(format!("pcmpestrm xmm0, xmm1, {imm8}").into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: b,
                    },
                    // Implicit argument to the pcmpestri intrinsic
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::ax)),
                        value: la,
                    },
                    // Implicit argument to the pcmpestri intrinsic
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::dx)),
                        value: lb,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.pclmulqdq" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_clmulepi64_si128&ig_expand=772
            intrinsic_args!(fx, args => (a, b, _imm8); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let imm8 =
                if let Some(imm8) = crate::constant::mir_operand_get_const_val(fx, &args[2].node) {
                    imm8
                } else {
                    fx.tcx.dcx().span_fatal(
                        span,
                        "Index argument for `_mm_clmulepi64_si128` is not a constant",
                    );
                };

            let imm8 = imm8.to_u8();

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(format!("pclmulqdq xmm0, xmm1, {imm8}").into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.aesni.aeskeygenassist" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aeskeygenassist_si128&ig_expand=261
            intrinsic_args!(fx, args => (a, _imm8); intrinsic);

            let a = a.load_scalar(fx);

            let imm8 =
                if let Some(imm8) = crate::constant::mir_operand_get_const_val(fx, &args[1].node) {
                    imm8
                } else {
                    fx.tcx.dcx().span_fatal(
                        span,
                        "Index argument for `_mm_aeskeygenassist_si128` is not a constant",
                    );
                };

            let imm8 = imm8.to_u8();

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(
                    format!("aeskeygenassist xmm0, xmm0, {imm8}").into(),
                )],
                &[CInlineAsmOperand::InOut {
                    reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                    _late: true,
                    in_value: a,
                    out_place: Some(ret),
                }],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.aesni.aesimc" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesimc_si128&ig_expand=260
            intrinsic_args!(fx, args => (a); intrinsic);

            let a = a.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("aesimc xmm0, xmm0".into())],
                &[CInlineAsmOperand::InOut {
                    reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                    _late: true,
                    in_value: a,
                    out_place: Some(ret),
                }],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.aesni.aesenc" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenc_si128&ig_expand=252
            intrinsic_args!(fx, args => (a, round_key); intrinsic);

            let a = a.load_scalar(fx);
            let round_key = round_key.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("aesenc xmm0, xmm1".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: round_key,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.aesni.aesenclast" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenclast_si128&ig_expand=257
            intrinsic_args!(fx, args => (a, round_key); intrinsic);

            let a = a.load_scalar(fx);
            let round_key = round_key.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("aesenclast xmm0, xmm1".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: round_key,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.aesni.aesdec" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdec_si128&ig_expand=242
            intrinsic_args!(fx, args => (a, round_key); intrinsic);

            let a = a.load_scalar(fx);
            let round_key = round_key.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("aesdec xmm0, xmm1".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: round_key,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.aesni.aesdeclast" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdeclast_si128&ig_expand=247
            intrinsic_args!(fx, args => (a, round_key); intrinsic);

            let a = a.load_scalar(fx);
            let round_key = round_key.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("aesdeclast xmm0, xmm1".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        value: round_key,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha1rnds4" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha1rnds4_epu32&ig_expand=5877
            intrinsic_args!(fx, args => (a, b, _func); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let func = if let Some(func) =
                crate::constant::mir_operand_get_const_val(fx, &args[2].node)
            {
                func
            } else {
                fx.tcx
                    .dcx()
                    .span_fatal(span, "Func argument for `_mm_sha1rnds4_epu32` is not a constant");
            };

            let func = func.to_u8();

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String(format!("sha1rnds4 xmm1, xmm2, {func}").into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha1msg1" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha1msg1_epu32&ig_expand=5874
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha1msg1 xmm1, xmm2".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha1msg2" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha1msg2_epu32&ig_expand=5875
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha1msg2 xmm1, xmm2".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha1nexte" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha1nexte_epu32&ig_expand=5876
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha1nexte xmm1, xmm2".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha256rnds2" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha256rnds2_epu32&ig_expand=5977
            intrinsic_args!(fx, args => (a, b, k); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);
            let k = k.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha256rnds2 xmm1, xmm2".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                    // Implicit argument to the sha256rnds2 instruction
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm0)),
                        value: k,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha256msg1" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha256msg1_epu32&ig_expand=5975
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha256msg1 xmm1, xmm2".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.sha256msg2" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha256msg2_epu32&ig_expand=5976
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("sha256msg2 xmm1, xmm2".into())],
                &[
                    CInlineAsmOperand::InOut {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm1)),
                        _late: true,
                        in_value: a,
                        out_place: Some(ret),
                    },
                    CInlineAsmOperand::In {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::xmm2)),
                        value: b,
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::PURE | InlineAsmOptions::NOMEM,
            );
        }

        "llvm.x86.avx.ptestz.256" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testz_si256&ig_expand=6945
            intrinsic_args!(fx, args => (a, b); intrinsic);

            assert_eq!(a.layout(), b.layout());
            let layout = a.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            assert_eq!(lane_ty, fx.tcx.types.i64);
            assert_eq!(ret.layout().ty, fx.tcx.types.i32);
            assert_eq!(lane_count, 4);

            let a_lane0 = a.value_lane(fx, 0).load_scalar(fx);
            let a_lane1 = a.value_lane(fx, 1).load_scalar(fx);
            let a_lane2 = a.value_lane(fx, 2).load_scalar(fx);
            let a_lane3 = a.value_lane(fx, 3).load_scalar(fx);
            let b_lane0 = b.value_lane(fx, 0).load_scalar(fx);
            let b_lane1 = b.value_lane(fx, 1).load_scalar(fx);
            let b_lane2 = b.value_lane(fx, 2).load_scalar(fx);
            let b_lane3 = b.value_lane(fx, 3).load_scalar(fx);

            let zero0 = fx.bcx.ins().band(a_lane0, b_lane0);
            let zero1 = fx.bcx.ins().band(a_lane1, b_lane1);
            let zero2 = fx.bcx.ins().band(a_lane2, b_lane2);
            let zero3 = fx.bcx.ins().band(a_lane3, b_lane3);

            let all_zero0 = fx.bcx.ins().bor(zero0, zero1);
            let all_zero1 = fx.bcx.ins().bor(zero2, zero3);
            let all_zero = fx.bcx.ins().bor(all_zero0, all_zero1);

            let res = fx.bcx.ins().icmp_imm(IntCC::Equal, all_zero, 0);
            let res = CValue::by_val(
                fx.bcx.ins().uextend(types::I32, res),
                fx.layout_of(fx.tcx.types.i32),
            );
            ret.write_cvalue(fx, res);
        }

        "llvm.x86.rdtsc" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_rdtsc&ig_expand=5273

            let res_place = CPlace::new_stack_slot(
                fx,
                fx.layout_of(Ty::new_tup(fx.tcx, &[fx.tcx.types.u32, fx.tcx.types.u32])),
            );
            let eax_place = res_place.place_field(fx, FieldIdx::new(0));
            let edx_place = res_place.place_field(fx, FieldIdx::new(1));
            codegen_inline_asm_inner(
                fx,
                &[InlineAsmTemplatePiece::String("rdtsc".into())],
                &[
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::ax)),
                        late: true,
                        place: Some(eax_place),
                    },
                    CInlineAsmOperand::Out {
                        reg: InlineAsmRegOrRegClass::Reg(InlineAsmReg::X86(X86InlineAsmReg::dx)),
                        late: true,
                        place: Some(edx_place),
                    },
                ],
                InlineAsmOptions::NOSTACK | InlineAsmOptions::NOMEM,
            );
            let res = res_place.to_cvalue(fx);
            ret.write_cvalue_transmute(fx, res);
        }

        _ => {
            fx.tcx
                .dcx()
                .warn(format!("unsupported x86 llvm intrinsic {}; replacing with trap", intrinsic));
            crate::trap::trap_unimplemented(fx, intrinsic);
            return;
        }
    }

    let dest = target.expect("all llvm intrinsics used by stdlib should return");
    let ret_block = fx.get_block(dest);
    fx.bcx.ins().jump(ret_block, &[]);
}

// llvm.x86.avx2.vperm2i128
// llvm.x86.ssse3.pshuf.b.128
// llvm.x86.avx2.pshuf.b

fn llvm_add_sub<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    cb_in: Value,
    a: CValue<'tcx>,
    b: CValue<'tcx>,
) -> (Value, Value) {
    assert_eq!(a.layout().ty, b.layout().ty);

    // c + carry -> c + first intermediate carry or borrow respectively
    let int0 = crate::num::codegen_checked_int_binop(fx, bin_op, a, b);
    let c = int0.value_field(fx, FieldIdx::ZERO);
    let cb0 = int0.value_field(fx, FieldIdx::new(1)).load_scalar(fx);

    // c + carry -> c + second intermediate carry or borrow respectively
    let clif_ty = fx.clif_type(a.layout().ty).unwrap();
    let cb_in_as_int = fx.bcx.ins().uextend(clif_ty, cb_in);
    let cb_in_as_int = CValue::by_val(cb_in_as_int, fx.layout_of(a.layout().ty));
    let int1 = crate::num::codegen_checked_int_binop(fx, bin_op, c, cb_in_as_int);
    let (c, cb1) = int1.load_scalar_pair(fx);

    // carry0 | carry1 -> carry or borrow respectively
    let cb_out = fx.bcx.ins().bor(cb0, cb1);

    (cb_out, c)
}

enum PackSize {
    U8,
    U16,
    S8,
    S16,
}

impl PackSize {
    fn ret_clif_type(&self) -> Type {
        match self {
            Self::U8 | Self::S8 => types::I8,
            Self::U16 | Self::S16 => types::I16,
        }
    }
    fn src_clif_type(&self) -> Type {
        match self {
            Self::U8 | Self::S8 => types::I16,
            Self::U16 | Self::S16 => types::I32,
        }
    }
    fn src_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self {
            Self::U8 | Self::S8 => tcx.types.i16,
            Self::U16 | Self::S16 => tcx.types.i32,
        }
    }
    fn ret_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self {
            Self::U8 => tcx.types.u8,
            Self::S8 => tcx.types.i8,
            Self::U16 => tcx.types.u16,
            Self::S16 => tcx.types.i16,
        }
    }
    fn max(&self) -> i64 {
        match self {
            Self::U8 => u8::MAX as u64 as i64,
            Self::S8 => i8::MAX as u8 as u64 as i64,
            Self::U16 => u16::MAX as u64 as i64,
            Self::S16 => i16::MAX as u64 as u64 as i64,
        }
    }
    fn min(&self) -> i64 {
        match self {
            Self::U8 | Self::U16 => 0,
            Self::S8 => i16::from(i8::MIN) as u16 as i64,
            Self::S16 => i32::from(i16::MIN) as u32 as i64,
        }
    }
}

enum PackWidth {
    Sse = 1,
    Avx = 2,
}
impl PackWidth {
    fn divisor(&self) -> u64 {
        match self {
            Self::Sse => 1,
            Self::Avx => 2,
        }
    }
}

/// Implement an x86 pack instruction with the intrinsic `_mm{,256}pack{us,s}_epi{16,32}`.
/// Validated for correctness against LLVM, see commit `c8f5d35508e062bd2d95e6c03429bfec831db6d3`.
fn pack_instruction<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    a: CValue<'tcx>,
    b: CValue<'tcx>,
    ret: CPlace<'tcx>,
    ret_size: PackSize,
    width: PackWidth,
) {
    assert_eq!(a.layout(), b.layout());
    let layout = a.layout();

    let (src_lane_count, src_lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    assert_eq!(src_lane_ty, ret_size.src_ty(fx.tcx));
    assert_eq!(ret_lane_ty, ret_size.ret_ty(fx.tcx));
    assert_eq!(src_lane_count * 2, ret_lane_count);

    let min = fx.bcx.ins().iconst(ret_size.src_clif_type(), ret_size.min());
    let max = fx.bcx.ins().iconst(ret_size.src_clif_type(), ret_size.max());
    let ret_lane_layout = fx.layout_of(ret_size.ret_ty(fx.tcx));

    let mut round = |source: CValue<'tcx>, source_offset: u64, dest_offset: u64| {
        let step_amount = src_lane_count / width.divisor();
        let dest_offset = step_amount * dest_offset;
        for idx in 0..step_amount {
            let lane = source.value_lane(fx, step_amount * source_offset + idx).load_scalar(fx);
            let sat = fx.bcx.ins().smax(lane, min);
            let sat = match ret_size {
                PackSize::U8 | PackSize::U16 => fx.bcx.ins().umin(sat, max),
                PackSize::S8 | PackSize::S16 => fx.bcx.ins().smin(sat, max),
            };
            let res = fx.bcx.ins().ireduce(ret_size.ret_clif_type(), sat);
            let res_lane = CValue::by_val(res, ret_lane_layout);
            ret.place_lane(fx, dest_offset + idx).write_cvalue(fx, res_lane);
        }
    };

    round(a, 0, 0);
    round(b, 0, 1);

    if let PackWidth::Avx = width {
        round(a, 1, 2);
        round(b, 1, 3);
    }
}
