//! Emulate x86 LLVM intrinsics

use crate::intrinsics::*;
use crate::prelude::*;

use rustc_middle::ty::subst::SubstsRef;

pub(crate) fn codegen_x86_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    _substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
) {
    match intrinsic {
        "llvm.x86.sse2.pause" | "llvm.aarch64.isb" => {
            // Spin loop hint
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
        "llvm.x86.sse2.cmp.ps" | "llvm.x86.sse2.cmp.pd" => {
            let (x, y, kind) = match args {
                [x, y, kind] => (x, y, kind),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let x = codegen_operand(fx, x);
            let y = codegen_operand(fx, y);
            let kind = crate::constant::mir_operand_get_const_val(fx, kind)
                .expect("llvm.x86.sse2.cmp.* kind not const");

            let flt_cc = match kind
                .try_to_bits(Size::from_bytes(1))
                .unwrap_or_else(|| panic!("kind not scalar: {:?}", kind))
            {
                0 => FloatCC::Equal,
                1 => FloatCC::LessThan,
                2 => FloatCC::LessThanOrEqual,
                7 => FloatCC::Ordered,
                3 => FloatCC::Unordered,
                4 => FloatCC::NotEqual,
                5 => FloatCC::UnorderedOrGreaterThanOrEqual,
                6 => FloatCC::UnorderedOrGreaterThan,
                kind => unreachable!("kind {:?}", kind),
            };

            simd_pair_for_each_lane(fx, x, y, ret, &|fx, lane_ty, res_lane_ty, x_lane, y_lane| {
                let res_lane = match lane_ty.kind() {
                    ty::Float(_) => fx.bcx.ins().fcmp(flt_cc, x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_ty),
                };
                bool_to_zero_or_max_uint(fx, res_lane_ty, res_lane)
            });
        }
        "llvm.x86.sse2.psrli.d" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.sse2.psrli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 32 => fx.bcx.ins().ushr_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.sse2.pslli.d" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.sse2.pslli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 32 => fx.bcx.ins().ishl_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.sse2.psrli.w" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.sse2.psrli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 16 => fx.bcx.ins().ushr_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.sse2.pslli.w" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.sse2.pslli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 16 => fx.bcx.ins().ishl_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.avx.psrli.d" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.avx.psrli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 32 => fx.bcx.ins().ushr_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.avx.pslli.d" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.avx.pslli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 32 => fx.bcx.ins().ishl_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.avx2.psrli.w" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.avx.psrli.w imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 16 => fx.bcx.ins().ushr_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        "llvm.x86.avx2.pslli.w" => {
            let (a, imm8) = match args {
                [a, imm8] => (a, imm8),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8)
                .expect("llvm.x86.avx.pslli.w imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 16 => fx.bcx.ins().ishl_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
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
        "llvm.x86.sse2.storeu.dq" => {
            intrinsic_args!(fx, args => (mem_addr, a); intrinsic);
            let mem_addr = mem_addr.load_scalar(fx);

            // FIXME correctly handle the unalignment
            let dest = CPlace::for_ptr(Pointer::new(mem_addr), a.layout());
            dest.write_cvalue(fx, a);
        }
        "llvm.x86.addcarry.64" => {
            intrinsic_args!(fx, args => (c_in, a, b); intrinsic);
            let c_in = c_in.load_scalar(fx);

            llvm_add_sub(fx, BinOp::Add, ret, c_in, a, b);
        }
        "llvm.x86.subborrow.64" => {
            intrinsic_args!(fx, args => (b_in, a, b); intrinsic);
            let b_in = b_in.load_scalar(fx);

            llvm_add_sub(fx, BinOp::Sub, ret, b_in, a, b);
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
// llvm.x86.avx2.psrli.w
// llvm.x86.sse2.psrli.w

fn llvm_add_sub<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    ret: CPlace<'tcx>,
    cb_in: Value,
    a: CValue<'tcx>,
    b: CValue<'tcx>,
) {
    assert_eq!(
        a.layout().ty,
        fx.tcx.types.u64,
        "llvm.x86.addcarry.64/llvm.x86.subborrow.64 second operand must be u64"
    );
    assert_eq!(
        b.layout().ty,
        fx.tcx.types.u64,
        "llvm.x86.addcarry.64/llvm.x86.subborrow.64 third operand must be u64"
    );

    // c + carry -> c + first intermediate carry or borrow respectively
    let int0 = crate::num::codegen_checked_int_binop(fx, bin_op, a, b);
    let c = int0.value_field(fx, FieldIdx::new(0));
    let cb0 = int0.value_field(fx, FieldIdx::new(1)).load_scalar(fx);

    // c + carry -> c + second intermediate carry or borrow respectively
    let cb_in_as_u64 = fx.bcx.ins().uextend(types::I64, cb_in);
    let cb_in_as_u64 = CValue::by_val(cb_in_as_u64, fx.layout_of(fx.tcx.types.u64));
    let int1 = crate::num::codegen_checked_int_binop(fx, bin_op, c, cb_in_as_u64);
    let (c, cb1) = int1.load_scalar_pair(fx);

    // carry0 | carry1 -> carry or borrow respectively
    let cb_out = fx.bcx.ins().bor(cb0, cb1);

    let layout = fx.layout_of(Ty::new_tup(fx.tcx, &[fx.tcx.types.u8, fx.tcx.types.u64]));
    let val = CValue::by_val_pair(cb_out, c, layout);
    ret.write_cvalue(fx, val);
}
