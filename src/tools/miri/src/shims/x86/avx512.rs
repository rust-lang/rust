use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{permute, pmaddbw, psadbw};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_avx512_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.avx512.").unwrap();

        match unprefixed_name {
            // Used by the ternarylogic functions.
            "pternlog.d.128" | "pternlog.d.256" | "pternlog.d.512" => {
                this.expect_target_feature_for_intrinsic(link_name, "avx512f")?;
                if matches!(unprefixed_name, "pternlog.d.128" | "pternlog.d.256") {
                    this.expect_target_feature_for_intrinsic(link_name, "avx512vl")?;
                }

                let [a, b, c, imm8] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                assert_eq!(dest.layout, a.layout);
                assert_eq!(dest.layout, b.layout);
                assert_eq!(dest.layout, c.layout);

                // The signatures of these operations are:
                //
                // ```
                // fn vpternlogd(a: i32x16, b: i32x16, c: i32x16, imm8: i32) -> i32x16;
                // fn vpternlogd256(a: i32x8, b: i32x8, c: i32x8, imm8: i32) -> i32x8;
                // fn vpternlogd128(a: i32x4, b: i32x4, c: i32x4, imm8: i32) -> i32x4;
                // ```
                //
                // The element type is always a 32-bit integer, the width varies.

                let (a, _a_len) = this.project_to_simd(a)?;
                let (b, _b_len) = this.project_to_simd(b)?;
                let (c, _c_len) = this.project_to_simd(c)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                // Compute one lane with ternary table.
                let tern = |xa: u32, xb: u32, xc: u32, imm: u32| -> u32 {
                    let mut out = 0u32;
                    // At each bit position, select bit from imm8 at index = (a << 2) | (b << 1) | c
                    for bit in 0..32 {
                        let ia = (xa >> bit) & 1;
                        let ib = (xb >> bit) & 1;
                        let ic = (xc >> bit) & 1;
                        let idx = (ia << 2) | (ib << 1) | ic;
                        let v = (imm >> idx) & 1;
                        out |= v << bit;
                    }
                    out
                };

                let imm8 = this.read_scalar(imm8)?.to_u32()? & 0xFF;
                for i in 0..dest_len {
                    let a_lane = this.project_index(&a, i)?;
                    let b_lane = this.project_index(&b, i)?;
                    let c_lane = this.project_index(&c, i)?;
                    let d_lane = this.project_index(&dest, i)?;

                    let va = this.read_scalar(&a_lane)?.to_u32()?;
                    let vb = this.read_scalar(&b_lane)?.to_u32()?;
                    let vc = this.read_scalar(&c_lane)?.to_u32()?;

                    let r = tern(va, vb, vc, imm8);
                    this.write_scalar(Scalar::from_u32(r), &d_lane)?;
                }
            }
            // Used to implement the _mm512_sad_epu8 function.
            "psad.bw.512" => {
                this.expect_target_feature_for_intrinsic(link_name, "avx512bw")?;

                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                psadbw(this, left, right, dest)?
            }
            // Used to implement the _mm512_maddubs_epi16 function.
            "pmaddubs.w.512" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                pmaddbw(this, left, right, dest)?;
            }
            // Used to implement the _mm512_permutexvar_epi32 function.
            "permvar.si.512" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                permute(this, left, right, dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
