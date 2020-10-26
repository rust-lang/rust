//! Emulate LLVM intrinsics

use crate::intrinsics::*;
use crate::prelude::*;

use rustc_middle::ty::subst::SubstsRef;

pub(crate) fn codegen_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    intrinsic: &str,
    substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
) {
    let ret = destination.unwrap().0;

    intrinsic_match! {
        fx, intrinsic, substs, args,
        _ => {
            fx.tcx.sess.warn(&format!("unsupported llvm intrinsic {}; replacing with trap", intrinsic));
            crate::trap::trap_unimplemented(fx, intrinsic);
        };

        // Used by `_mm_movemask_epi8` and `_mm256_movemask_epi8`
        llvm.x86.sse2.pmovmskb.128 | llvm.x86.avx2.pmovmskb | llvm.x86.sse2.movmsk.pd, (c a) {
            let (lane_layout, lane_count) = lane_type_and_count(fx.tcx, a.layout());
            let lane_ty = fx.clif_type(lane_layout.ty).unwrap();
            assert!(lane_count <= 32);

            let mut res = fx.bcx.ins().iconst(types::I32, 0);

            for lane in (0..lane_count).rev() {
                let a_lane = a.value_field(fx, mir::Field::new(lane.try_into().unwrap())).load_scalar(fx);

                // cast float to int
                let a_lane = match lane_ty {
                    types::F32 => fx.bcx.ins().bitcast(types::I32, a_lane),
                    types::F64 => fx.bcx.ins().bitcast(types::I64, a_lane),
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
        };
        llvm.x86.sse2.cmp.ps | llvm.x86.sse2.cmp.pd, (c x, c y, o kind) {
            let kind_const = crate::constant::mir_operand_get_const_val(fx, kind).expect("llvm.x86.sse2.cmp.* kind not const");
            let flt_cc = match kind_const.val.try_to_bits(Size::from_bytes(1)).expect(&format!("kind not scalar: {:?}", kind_const)) {
                0 => FloatCC::Equal,
                1 => FloatCC::LessThan,
                2 => FloatCC::LessThanOrEqual,
                7 => {
                    unimplemented!("Compares corresponding elements in `a` and `b` to see if neither is `NaN`.");
                }
                3 => {
                    unimplemented!("Compares corresponding elements in `a` and `b` to see if either is `NaN`.");
                }
                4 => FloatCC::NotEqual,
                5 => {
                    unimplemented!("not less than");
                }
                6 => {
                    unimplemented!("not less than or equal");
                }
                kind => unreachable!("kind {:?}", kind),
            };

            simd_pair_for_each_lane(fx, x, y, ret, |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind() {
                    ty::Float(_) => fx.bcx.ins().fcmp(flt_cc, x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                bool_to_zero_or_max_uint(fx, res_lane_layout, res_lane)
            });
        };
        llvm.x86.sse2.psrli.d, (c a, o imm8) {
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8).expect("llvm.x86.sse2.psrli.d imm8 not const");
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, res_lane_layout, lane| {
                let res_lane = match imm8.val.try_to_bits(Size::from_bytes(4)).expect(&format!("imm8 not scalar: {:?}", imm8)) {
                    imm8 if imm8 < 32 => fx.bcx.ins().ushr_imm(lane, i64::from(imm8 as u8)),
                    _ => fx.bcx.ins().iconst(types::I32, 0),
                };
                CValue::by_val(res_lane, res_lane_layout)
            });
        };
        llvm.x86.sse2.pslli.d, (c a, o imm8) {
            let imm8 = crate::constant::mir_operand_get_const_val(fx, imm8).expect("llvm.x86.sse2.psrli.d imm8 not const");
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, res_lane_layout, lane| {
                let res_lane = match imm8.val.try_to_bits(Size::from_bytes(4)).expect(&format!("imm8 not scalar: {:?}", imm8)) {
                    imm8 if imm8 < 32 => fx.bcx.ins().ishl_imm(lane, i64::from(imm8 as u8)),
                    _ => fx.bcx.ins().iconst(types::I32, 0),
                };
                CValue::by_val(res_lane, res_lane_layout)
            });
        };
        llvm.x86.sse2.storeu.dq, (v mem_addr, c a) {
            // FIXME correctly handle the unalignment
            let dest = CPlace::for_ptr(Pointer::new(mem_addr), a.layout());
            dest.write_cvalue(fx, a);
        };
    }

    if let Some((_, dest)) = destination {
        let ret_block = fx.get_block(dest);
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}

// llvm.x86.avx2.vperm2i128
// llvm.x86.ssse3.pshuf.b.128
// llvm.x86.avx2.pshuf.b
// llvm.x86.avx2.psrli.w
// llvm.x86.sse2.psrli.w
