use crate::prelude::*;

use rustc::ty::subst::SubstsRef;

pub fn codegen_llvm_intrinsic_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    intrinsic: &str,
    substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
) {
    let ret = match destination {
        Some((place, _)) => place,
        None => {
            // Insert non returning intrinsics here
            match intrinsic {
                "abort" => {
                    trap_panic(fx, "Called intrinsic::abort.");
                }
                "unreachable" => {
                    trap_unreachable(fx, "[corruption] Called intrinsic::unreachable.");
                }
                _ => unimplemented!("unsupported instrinsic {}", intrinsic),
            }
            return;
        }
    };

    crate::intrinsics::intrinsic_match! {
        fx, intrinsic, substs, args,
        _ => {
            fx.tcx.sess.warn(&format!("unsupported llvm intrinsic {}; replacing with trap", intrinsic));
            crate::trap::trap_unimplemented(fx, intrinsic);
        };

        // Used by _mm_movemask_epi8
        llvm.x86.sse2.pmovmskb.128, (c a) {
            let (lane_layout, lane_count) = crate::intrinsics::lane_type_and_count(fx, a.layout(), intrinsic);
            assert_eq!(lane_layout.ty.sty, fx.tcx.types.i8.sty);
            assert_eq!(lane_count, 16);

            let mut res = fx.bcx.ins().iconst(types::I32, 0);

            for lane in 0..16 {
                let a_lane = a.value_field(fx, mir::Field::new(lane.try_into().unwrap())).load_scalar(fx);
                let a_lane_sign = fx.bcx.ins().ushr_imm(a_lane, 7); // extract sign bit of 8bit int
                let a_lane_sign = fx.bcx.ins().uextend(types::I32, a_lane_sign);
                res = fx.bcx.ins().ishl_imm(res, 1);
                res = fx.bcx.ins().bor(res, a_lane_sign);
            }

            let res = CValue::by_val(res, fx.layout_of(fx.tcx.types.i32));
            ret.write_cvalue(fx, res);
        };
    }

    if let Some((_, dest)) = destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}

// llvm.x86.avx2.vperm2i128
// llvm.x86.ssse3.pshuf.b.128
// llvm.x86.avx2.pshuf.b
// llvm.x86.avx2.pmovmskb
// llvm.x86.avx2.psrli.w
// llvm.x86.sse2.psrli.w
