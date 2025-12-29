use rustc_abi::CanonAbi;
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{
    ShiftOp, horizontal_bin_op, mpsadbw, packssdw, packsswb, packusdw, packuswb, permute, pmaddbw,
    pmulhrsw, psadbw, pshufb, psign, shift_simd_by_scalar,
};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_avx2_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "avx2")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.avx2.").unwrap();

        match unprefixed_name {
            // Used to implement the _mm256_h{adds,subs}_epi16 functions.
            // Horizontally add / subtract with saturation adjacent 16-bit
            // integer values in `left` and `right`.
            "phadd.sw" | "phsub.sw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "phadd.sw" => mir::BinOp::Add,
                    "phsub.sw" => mir::BinOp::Sub,
                    _ => unreachable!(),
                };

                horizontal_bin_op(this, which, /*saturating*/ true, left, right, dest)?;
            }
            // Used to implement `_mm{,_mask}_{i32,i64}gather_{epi32,epi64,pd,ps}` functions
            // Gathers elements from `slice` using `offsets * scale` as indices.
            // When the highest bit of the corresponding element of `mask` is 0,
            // the value is copied from `src` instead.
            "gather.d.d" | "gather.d.d.256" | "gather.d.q" | "gather.d.q.256" | "gather.q.d"
            | "gather.q.d.256" | "gather.q.q" | "gather.q.q.256" | "gather.d.pd"
            | "gather.d.pd.256" | "gather.q.pd" | "gather.q.pd.256" | "gather.d.ps"
            | "gather.d.ps.256" | "gather.q.ps" | "gather.q.ps.256" => {
                let [src, slice, offsets, mask, scale] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                assert_eq!(dest.layout, src.layout);

                let (src, _) = this.project_to_simd(src)?;
                let (offsets, offsets_len) = this.project_to_simd(offsets)?;
                let (mask, mask_len) = this.project_to_simd(mask)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                // There are cases like dest: i32x4, offsets: i64x2
                // If dest has more elements than offset, extra dest elements are filled with zero.
                // If offsets has more elements than dest, extra offsets are ignored.
                let actual_len = dest_len.min(offsets_len);

                assert_eq!(dest_len, mask_len);

                let mask_item_size = mask.layout.field(this, 0).size;
                let high_bit_offset = mask_item_size.bits().strict_sub(1);

                let scale = this.read_scalar(scale)?.to_i8()?;
                if !matches!(scale, 1 | 2 | 4 | 8) {
                    panic!("invalid gather scale {scale}");
                }
                let scale = i64::from(scale);

                let slice = this.read_pointer(slice)?;
                for i in 0..actual_len {
                    let mask = this.project_index(&mask, i)?;
                    let dest = this.project_index(&dest, i)?;

                    if this.read_scalar(&mask)?.to_uint(mask_item_size)? >> high_bit_offset != 0 {
                        let offset = this.project_index(&offsets, i)?;
                        let offset =
                            i64::try_from(this.read_scalar(&offset)?.to_int(offset.layout.size)?)
                                .unwrap();
                        let ptr = slice.wrapping_signed_offset(offset.strict_mul(scale), &this.tcx);
                        // Unaligned copy, which is what we want.
                        this.mem_copy(
                            ptr,
                            dest.ptr(),
                            dest.layout.size,
                            /*nonoverlapping*/ true,
                        )?;
                    } else {
                        this.copy_op(&this.project_index(&src, i)?, &dest)?;
                    }
                }
                for i in actual_len..dest_len {
                    let dest = this.project_index(&dest, i)?;
                    this.write_scalar(Scalar::from_int(0, dest.layout.size), &dest)?;
                }
            }
            // Used to implement the _mm256_maddubs_epi16 function.
            "pmadd.ub.sw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                pmaddbw(this, left, right, dest)?;
            }
            // Used to implement the _mm256_mpsadbw_epu8 function.
            // Compute the sum of absolute differences of quadruplets of unsigned
            // 8-bit integers in `left` and `right`, and store the 16-bit results
            // in `right`. Quadruplets are selected from `left` and `right` with
            // offsets specified in `imm`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mpsadbw_epu8
            "mpsadbw" => {
                let [left, right, imm] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                mpsadbw(this, left, right, imm, dest)?;
            }
            // Used to implement the _mm256_mulhrs_epi16 function.
            // Multiplies packed 16-bit signed integer values, truncates the 32-bit
            // product to the 18 most significant bits by right-shifting, and then
            // divides the 18-bit value by 2 (rounding to nearest) by first adding
            // 1 and then taking the bits `1..=16`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mulhrs_epi16
            "pmul.hr.sw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                pmulhrsw(this, left, right, dest)?;
            }
            // Used to implement the _mm256_packs_epi16 function.
            // Converts two 16-bit integer vectors to a single 8-bit integer
            // vector with signed saturation.
            "packsswb" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                packsswb(this, left, right, dest)?;
            }
            // Used to implement the _mm256_packs_epi32 function.
            // Converts two 32-bit integer vectors to a single 16-bit integer
            // vector with signed saturation.
            "packssdw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                packssdw(this, left, right, dest)?;
            }
            // Used to implement the _mm256_packus_epi16 function.
            // Converts two 16-bit signed integer vectors to a single 8-bit
            // unsigned integer vector with saturation.
            "packuswb" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                packuswb(this, left, right, dest)?;
            }
            // Used to implement the _mm256_packus_epi32 function.
            // Concatenates two 32-bit signed integer vectors and converts
            // the result to a 16-bit unsigned integer vector with saturation.
            "packusdw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                packusdw(this, left, right, dest)?;
            }
            // Used to implement _mm256_permutevar8x32_epi32 and _mm256_permutevar8x32_ps.
            "permd" | "permps" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                permute(this, left, right, dest)?;
            }
            // Used to implement the _mm256_sad_epu8 function.
            "psad.bw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                psadbw(this, left, right, dest)?
            }
            // Used to implement the _mm256_shuffle_epi8 intrinsic.
            // Shuffles bytes from `left` using `right` as pattern.
            // Each 128-bit block is shuffled independently.
            "pshuf.b" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                pshufb(this, left, right, dest)?;
            }
            // Used to implement the _mm256_sign_epi{8,16,32} functions.
            // Negates elements from `left` when the corresponding element in
            // `right` is negative. If an element from `right` is zero, zero
            // is writen to the corresponding output element.
            // Basically, we multiply `left` with `right.signum()`.
            "psign.b" | "psign.w" | "psign.d" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                psign(this, left, right, dest)?;
            }
            // Used to implement the _mm256_{sll,srl,sra}_epi{16,32,64} functions
            // (except _mm256_sra_epi64, which is not available in AVX2).
            // Shifts N-bit packed integers in left by the amount in right.
            // `right` is as 128-bit vector. but it is interpreted as a single
            // 64-bit integer (remaining bits are ignored).
            // For logic shifts, when right is larger than N - 1, zero is produced.
            // For arithmetic shifts, when right is larger than N - 1, the sign bit
            // is copied to remaining bits.
            "psll.w" | "psrl.w" | "psra.w" | "psll.d" | "psrl.d" | "psra.d" | "psll.q"
            | "psrl.q" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "psll.w" | "psll.d" | "psll.q" => ShiftOp::Left,
                    "psrl.w" | "psrl.d" | "psrl.q" => ShiftOp::RightLogic,
                    "psra.w" | "psra.d" => ShiftOp::RightArith,
                    _ => unreachable!(),
                };

                shift_simd_by_scalar(this, left, right, which, dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
