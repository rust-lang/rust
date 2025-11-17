use rustc_abi::CanonAbi;
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{horizontal_bin_op, pmaddbw, pmulhrsw, psign};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_ssse3_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "ssse3")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.ssse3.").unwrap();

        match unprefixed_name {
            // Used to implement the _mm_shuffle_epi8 intrinsic.
            // Shuffles bytes from `left` using `right` as pattern.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_epi8
            "pshuf.b.128" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_u8()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = if right & 0x80 == 0 {
                        let j = right % 16; // index wraps around
                        this.read_scalar(&this.project_index(&left, j.into())?)?
                    } else {
                        // If the highest bit in `right` is 1, write zero.
                        Scalar::from_u8(0)
                    };

                    this.write_scalar(res, &dest)?;
                }
            }
            // Used to implement the _mm_h{adds,subs}_epi16 functions.
            // Horizontally add / subtract with saturation adjacent 16-bit
            // integer values in `left` and `right`.
            "phadd.sw.128" | "phsub.sw.128" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "phadd.sw.128" => mir::BinOp::Add,
                    "phsub.sw.128" => mir::BinOp::Sub,
                    _ => unreachable!(),
                };

                horizontal_bin_op(this, which, /*saturating*/ true, left, right, dest)?;
            }
            // Used to implement the _mm_maddubs_epi16 function.
            "pmadd.ub.sw.128" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                pmaddbw(this, left, right, dest)?;
            }
            // Used to implement the _mm_mulhrs_epi16 function.
            // Multiplies packed 16-bit signed integer values, truncates the 32-bit
            // product to the 18 most significant bits by right-shifting, and then
            // divides the 18-bit value by 2 (rounding to nearest) by first adding
            // 1 and then taking the bits `1..=16`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mulhrs_epi16
            "pmul.hr.sw.128" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                pmulhrsw(this, left, right, dest)?;
            }
            // Used to implement the _mm_sign_epi{8,16,32} functions.
            // Negates elements from `left` when the corresponding element in
            // `right` is negative. If an element from `right` is zero, zero
            // is writen to the corresponding output element.
            // Basically, we multiply `left` with `right.signum()`.
            "psign.b.128" | "psign.w.128" | "psign.d.128" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                psign(this, left, right, dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
