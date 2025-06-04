use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{conditional_dot_product, mpsadbw, packusdw, round_all, round_first, test_bits_masked};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sse41_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sse4.1")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse41.").unwrap();

        match unprefixed_name {
            // Used to implement the _mm_insert_ps function.
            // Takes one element of `right` and inserts it into `left` and
            // optionally zero some elements. Source index is specified
            // in bits `6..=7` of `imm`, destination index is specified in
            // bits `4..=5` if `imm`, and `i`th bit specifies whether element
            // `i` is zeroed.
            "insertps" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);
                assert!(dest_len <= 4);

                let imm = this.read_scalar(imm)?.to_u8()?;
                let src_index = u64::from((imm >> 6) & 0b11);
                let dst_index = u64::from((imm >> 4) & 0b11);

                let src_value = this.read_immediate(&this.project_index(&right, src_index)?)?;

                for i in 0..dest_len {
                    let dest = this.project_index(&dest, i)?;

                    if imm & (1 << i) != 0 {
                        // zeroed
                        this.write_scalar(Scalar::from_u32(0), &dest)?;
                    } else if i == dst_index {
                        // copy from `right` at specified index
                        this.write_immediate(*src_value, &dest)?;
                    } else {
                        // copy from `left`
                        this.copy_op(&this.project_index(&left, i)?, &dest)?;
                    }
                }
            }
            // Used to implement the _mm_packus_epi32 function.
            // Concatenates two 32-bit signed integer vectors and converts
            // the result to a 16-bit unsigned integer vector with saturation.
            "packusdw" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                packusdw(this, left, right, dest)?;
            }
            // Used to implement the _mm_dp_ps and _mm_dp_pd functions.
            // Conditionally multiplies the packed floating-point elements in
            // `left` and `right` using the high 4 bits in `imm`, sums the four
            // products, and conditionally stores the sum in `dest` using the low
            // 4 bits of `imm`.
            "dpps" | "dppd" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                conditional_dot_product(this, left, right, imm, dest)?;
            }
            // Used to implement the _mm_floor_ss, _mm_ceil_ss and _mm_round_ss
            // functions. Rounds the first element of `right` according to `rounding`
            // and copies the remaining elements from `left`.
            "round.ss" => {
                let [left, right, rounding] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                round_first::<rustc_apfloat::ieee::Single>(this, left, right, rounding, dest)?;
            }
            // Used to implement the _mm_floor_ps, _mm_ceil_ps and _mm_round_ps
            // functions. Rounds the elements of `op` according to `rounding`.
            "round.ps" => {
                let [op, rounding] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                round_all::<rustc_apfloat::ieee::Single>(this, op, rounding, dest)?;
            }
            // Used to implement the _mm_floor_sd, _mm_ceil_sd and _mm_round_sd
            // functions. Rounds the first element of `right` according to `rounding`
            // and copies the remaining elements from `left`.
            "round.sd" => {
                let [left, right, rounding] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                round_first::<rustc_apfloat::ieee::Double>(this, left, right, rounding, dest)?;
            }
            // Used to implement the _mm_floor_pd, _mm_ceil_pd and _mm_round_pd
            // functions. Rounds the elements of `op` according to `rounding`.
            "round.pd" => {
                let [op, rounding] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                round_all::<rustc_apfloat::ieee::Double>(this, op, rounding, dest)?;
            }
            // Used to implement the _mm_minpos_epu16 function.
            // Find the minimum unsinged 16-bit integer in `op` and
            // returns its value and position.
            "phminposuw" => {
                let [op] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                // Find minimum
                let mut min_value = u16::MAX;
                let mut min_index = 0;
                for i in 0..op_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_u16()?;
                    if op < min_value {
                        min_value = op;
                        min_index = i;
                    }
                }

                // Write value and index
                this.write_scalar(Scalar::from_u16(min_value), &this.project_index(&dest, 0)?)?;
                this.write_scalar(
                    Scalar::from_u16(min_index.try_into().unwrap()),
                    &this.project_index(&dest, 1)?,
                )?;
                // Fill remainder with zeros
                for i in 2..dest_len {
                    this.write_scalar(Scalar::from_u16(0), &this.project_index(&dest, i)?)?;
                }
            }
            // Used to implement the _mm_mpsadbw_epu8 function.
            // Compute the sum of absolute differences of quadruplets of unsigned
            // 8-bit integers in `left` and `right`, and store the 16-bit results
            // in `right`. Quadruplets are selected from `left` and `right` with
            // offsets specified in `imm`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mpsadbw_epu8
            "mpsadbw" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                mpsadbw(this, left, right, imm, dest)?;
            }
            // Used to implement the _mm_testz_si128, _mm_testc_si128
            // and _mm_testnzc_si128 functions.
            // Tests `(op & mask) == 0`, `(op & mask) == mask` or
            // `(op & mask) != 0 && (op & mask) != mask`
            "ptestz" | "ptestc" | "ptestnzc" => {
                let [op, mask] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (all_zero, masked_set) = test_bits_masked(this, op, mask)?;
                let res = match unprefixed_name {
                    "ptestz" => all_zero,
                    "ptestc" => masked_set,
                    "ptestnzc" => !all_zero && !masked_set,
                    _ => unreachable!(),
                };

                this.write_scalar(Scalar::from_i32(res.into()), dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
