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
        "llvm.x86.sse2.pmovmskb.128" | "llvm.x86.avx2.pmovmskb" | "llvm.x86.sse2.movmsk.pd" => {
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
                .expect("llvm.x86.sse2.psrli.d imm8 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm8
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm8 not scalar: {:?}", imm8))
            {
                imm8 if imm8 < 32 => fx.bcx.ins().ishl_imm(lane, i64::from(imm8 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
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
            fx.tcx.sess.warn(&format!(
                "unsupported x86 llvm intrinsic {}; replacing with trap",
                intrinsic
            ));
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
    let c = int0.value_field(fx, mir::Field::new(0));
    let cb0 = int0.value_field(fx, mir::Field::new(1)).load_scalar(fx);

    // c + carry -> c + second intermediate carry or borrow respectively
    let cb_in_as_u64 = fx.bcx.ins().uextend(types::I64, cb_in);
    let cb_in_as_u64 = CValue::by_val(cb_in_as_u64, fx.layout_of(fx.tcx.types.u64));
    let int1 = crate::num::codegen_checked_int_binop(fx, bin_op, c, cb_in_as_u64);
    let (c, cb1) = int1.load_scalar_pair(fx);

    // carry0 | carry1 -> carry or borrow respectively
    let cb_out = fx.bcx.ins().bor(cb0, cb1);

    let layout = fx.layout_of(fx.tcx.intern_tup(&[fx.tcx.types.u8, fx.tcx.types.u64]));
    let val = CValue::by_val_pair(cb_out, c, layout);
    ret.write_cvalue(fx, val);
}
