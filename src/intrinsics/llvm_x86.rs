//! Emulate x86 LLVM intrinsics

use rustc_middle::ty::GenericArgsRef;

use crate::intrinsics::*;
use crate::prelude::*;

pub(crate) fn codegen_x86_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    _args: GenericArgsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
) {
    match intrinsic {
        "llvm.x86.sse2.pause" | "llvm.aarch64.isb" => {
            // Spin loop hint
        }

        // Used by is_x86_feature_detected!();
        "llvm.x86.xgetbv" => {
            // FIXME use the actual xgetbv instruction
            intrinsic_args!(fx, args => (v); intrinsic);

            let v = v.load_scalar(fx);

            // As of writing on XCR0 exists
            fx.bcx.ins().trapnz(v, TrapCode::UnreachableCodeReached);

            let res = fx.bcx.ins().iconst(types::I64, 1 /* bit 0 must be set */);
            ret.write_cvalue(fx, CValue::by_val(res, fx.layout_of(fx.tcx.types.i64)));
        }

        // Used by `_mm_movemask_epi8` and `_mm256_movemask_epi8`
        "llvm.x86.sse2.pmovmskb.128"
        | "llvm.x86.avx2.pmovmskb"
        | "llvm.x86.sse.movmsk.ps"
        | "llvm.x86.sse2.movmsk.pd" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            let (lane_count, lane_ty) = a.layout().ty.simd_size_and_type(fx.tcx);
            let lane_ty = fx.clif_type(lane_ty).unwrap();
            assert!(lane_count <= 32);

            let mut res = fx.bcx.ins().iconst(types::I32, 0);

            for lane in (0..lane_count).rev() {
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);

                // cast float to int
                let a_lane = match lane_ty {
                    types::F32 => codegen_bitcast(fx, types::I32, a_lane),
                    types::F64 => codegen_bitcast(fx, types::I64, a_lane),
                    _ => a_lane,
                };

                // extract sign bit of an int
                let a_lane_sign = fx.bcx.ins().ushr_imm(a_lane, i64::from(lane_ty.bits() - 1));

                // shift sign bit into result
                let a_lane_sign = clif_intcast(fx, a_lane_sign, types::I32, false);
                res = fx.bcx.ins().ishl_imm(res, 1);
                res = fx.bcx.ins().bor(res, a_lane_sign);
            }

            let res = CValue::by_val(res, fx.layout_of(fx.tcx.types.i32));
            ret.write_cvalue(fx, res);
        }
        "llvm.x86.sse.cmp.ps" | "llvm.x86.sse2.cmp.pd" => {
            let (x, y, kind) = match args {
                [x, y, kind] => (x, y, kind),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let x = codegen_operand(fx, x);
            let y = codegen_operand(fx, y);
            let kind = match kind {
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
            let a = codegen_operand(fx, a);
            let b = codegen_operand(fx, b);

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
        "llvm.x86.avx2.vperm2i128" => {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permute2x128_si256
            let (a, b, imm8) = match args {
                [a, b, imm8] => (a, b, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let b = codegen_operand(fx, b);
            let imm8 = codegen_operand(fx, imm8).load_scalar(fx);

            let a_0 = a.value_lane(fx, 0).load_scalar(fx);
            let a_1 = a.value_lane(fx, 1).load_scalar(fx);
            let a_low = fx.bcx.ins().iconcat(a_0, a_1);
            let a_2 = a.value_lane(fx, 2).load_scalar(fx);
            let a_3 = a.value_lane(fx, 3).load_scalar(fx);
            let a_high = fx.bcx.ins().iconcat(a_2, a_3);

            let b_0 = b.value_lane(fx, 0).load_scalar(fx);
            let b_1 = b.value_lane(fx, 1).load_scalar(fx);
            let b_low = fx.bcx.ins().iconcat(b_0, b_1);
            let b_2 = b.value_lane(fx, 2).load_scalar(fx);
            let b_3 = b.value_lane(fx, 3).load_scalar(fx);
            let b_high = fx.bcx.ins().iconcat(b_2, b_3);

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
            let (res_0, res_1) = fx.bcx.ins().isplit(res_low);

            let control1 = fx.bcx.ins().ushr_imm(imm8, 4);
            let res_high = select4(fx, a_high, a_low, b_high, b_low, control1);
            let (res_2, res_3) = fx.bcx.ins().isplit(res_high);

            ret.place_lane(fx, 0).to_ptr().store(fx, res_0, MemFlags::trusted());
            ret.place_lane(fx, 1).to_ptr().store(fx, res_1, MemFlags::trusted());
            ret.place_lane(fx, 2).to_ptr().store(fx, res_2, MemFlags::trusted());
            ret.place_lane(fx, 3).to_ptr().store(fx, res_3, MemFlags::trusted());
        }
        "llvm.x86.ssse3.pabs.b.128" | "llvm.x86.ssse3.pabs.w.128" | "llvm.x86.ssse3.pabs.d.128" => {
            let a = match args {
                [a] => a,
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().iabs(lane)
            });
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
        "llvm.x86.sse2.psad.bw" => {
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

                for lane_idx in out_lane_idx * 8..out_lane_idx * 8 + 1 {
                    let a_lane = a.value_lane(fx, lane_idx).load_scalar(fx);
                    let b_lane = b.value_lane(fx, lane_idx).load_scalar(fx);

                    let lane_diff = fx.bcx.ins().isub(a_lane, b_lane);
                    let abs_lane_diff = fx.bcx.ins().iabs(lane_diff);
                    let abs_lane_diff = fx.bcx.ins().uextend(types::I64, abs_lane_diff);
                    lane_diff_acc = fx.bcx.ins().iadd(lane_diff_acc, abs_lane_diff);
                }

                let res_lane = CValue::by_val(lane_diff_acc, ret_lane_layout);

                ret.place_lane(fx, out_lane_idx).write_cvalue(fx, res_lane);
            }
        }
        "llvm.x86.ssse3.pmadd.ub.sw.128" => {
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
        "llvm.x86.sse2.pmadd.wd" => {
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
                let a_lane0 = fx.bcx.ins().uextend(types::I32, a_lane0);
                let b_lane0 = b.value_lane(fx, out_lane_idx * 2).load_scalar(fx);
                let b_lane0 = fx.bcx.ins().sextend(types::I32, b_lane0);

                let a_lane1 = a.value_lane(fx, out_lane_idx * 2 + 1).load_scalar(fx);
                let a_lane1 = fx.bcx.ins().uextend(types::I32, a_lane1);
                let b_lane1 = b.value_lane(fx, out_lane_idx * 2 + 1).load_scalar(fx);
                let b_lane1 = fx.bcx.ins().sextend(types::I32, b_lane1);

                let mul0: Value = fx.bcx.ins().imul(a_lane0, b_lane0);
                let mul1 = fx.bcx.ins().imul(a_lane1, b_lane1);

                let res_lane = fx.bcx.ins().iadd(mul0, mul1);
                let res_lane = CValue::by_val(res_lane, ret_lane_layout);

                ret.place_lane(fx, out_lane_idx).write_cvalue(fx, res_lane);
            }
        }
        _ => {
            fx.tcx
                .sess
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
    let c = int0.value_field(fx, FieldIdx::new(0));
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
