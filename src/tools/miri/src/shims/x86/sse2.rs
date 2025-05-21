use rustc_abi::CanonAbi;
use rustc_apfloat::ieee::Double;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{
    FloatBinOp, ShiftOp, bin_op_simd_float_all, bin_op_simd_float_first, convert_float_to_int,
    packssdw, packsswb, packuswb, shift_simd_by_scalar,
};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sse2_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sse2")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse2.").unwrap();

        // These intrinsics operate on 128-bit (f32x4, f64x2, i8x16, i16x8, i32x4, i64x2) SIMD
        // vectors unless stated otherwise.
        // Many intrinsic names are sufixed with "ps" (packed single), "ss" (scalar signle),
        // "pd" (packed double) or "sd" (scalar double), where single means single precision
        // floating point (f32) and double means double precision floating point (f64). "ps"
        // and "pd" means thet the operation is performed on each element of the vector, while
        // "ss" and "sd" means that the operation is performed only on the first element, copying
        // the remaining elements from the input vector (for binary operations, from the left-hand
        // side).
        // Intrinsincs sufixed with "epiX" or "epuX" operate with X-bit signed or unsigned
        // vectors.
        match unprefixed_name {
            // Used to implement the _mm_madd_epi16 function.
            // Multiplies packed signed 16-bit integers in `left` and `right`, producing
            // intermediate signed 32-bit integers. Horizontally add adjacent pairs of
            // intermediate 32-bit integers, and pack the results in `dest`.
            "pmadd.wd" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(dest_len.strict_mul(2), left_len);

                for i in 0..dest_len {
                    let j1 = i.strict_mul(2);
                    let left1 = this.read_scalar(&this.project_index(&left, j1)?)?.to_i16()?;
                    let right1 = this.read_scalar(&this.project_index(&right, j1)?)?.to_i16()?;

                    let j2 = j1.strict_add(1);
                    let left2 = this.read_scalar(&this.project_index(&left, j2)?)?.to_i16()?;
                    let right2 = this.read_scalar(&this.project_index(&right, j2)?)?.to_i16()?;

                    let dest = this.project_index(&dest, i)?;

                    // Multiplications are i16*i16->i32, which will not overflow.
                    let mul1 = i32::from(left1).strict_mul(right1.into());
                    let mul2 = i32::from(left2).strict_mul(right2.into());
                    // However, this addition can overflow in the most extreme case
                    // (-0x8000)*(-0x8000)+(-0x8000)*(-0x8000) = 0x80000000
                    let res = mul1.wrapping_add(mul2);

                    this.write_scalar(Scalar::from_i32(res), &dest)?;
                }
            }
            // Used to implement the _mm_sad_epu8 function.
            // Computes the absolute differences of packed unsigned 8-bit integers in `a`
            // and `b`, then horizontally sum each consecutive 8 differences to produce
            // two unsigned 16-bit integers, and pack these unsigned 16-bit integers in
            // the low 16 bits of 64-bit elements returned.
            //
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sad_epu8
            "psad.bw" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                // left and right are u8x16, dest is u64x2
                assert_eq!(left_len, right_len);
                assert_eq!(left_len, 16);
                assert_eq!(dest_len, 2);

                for i in 0..dest_len {
                    let dest = this.project_index(&dest, i)?;

                    let mut res: u16 = 0;
                    let n = left_len.strict_div(dest_len);
                    for j in 0..n {
                        let op_i = j.strict_add(i.strict_mul(n));
                        let left = this.read_scalar(&this.project_index(&left, op_i)?)?.to_u8()?;
                        let right =
                            this.read_scalar(&this.project_index(&right, op_i)?)?.to_u8()?;

                        res = res.strict_add(left.abs_diff(right).into());
                    }

                    this.write_scalar(Scalar::from_u64(res.into()), &dest)?;
                }
            }
            // Used to implement the _mm_{sll,srl,sra}_epi{16,32,64} functions
            // (except _mm_sra_epi64, which is not available in SSE2).
            // Shifts N-bit packed integers in left by the amount in right.
            // Both operands are 128-bit vectors. However, right is interpreted as
            // a single 64-bit integer (remaining bits are ignored).
            // For logic shifts, when right is larger than N - 1, zero is produced.
            // For arithmetic shifts, when right is larger than N - 1, the sign bit
            // is copied to remaining bits.
            "psll.w" | "psrl.w" | "psra.w" | "psll.d" | "psrl.d" | "psra.d" | "psll.q"
            | "psrl.q" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "psll.w" | "psll.d" | "psll.q" => ShiftOp::Left,
                    "psrl.w" | "psrl.d" | "psrl.q" => ShiftOp::RightLogic,
                    "psra.w" | "psra.d" => ShiftOp::RightArith,
                    _ => unreachable!(),
                };

                shift_simd_by_scalar(this, left, right, which, dest)?;
            }
            // Used to implement the _mm_cvtps_epi32, _mm_cvttps_epi32, _mm_cvtpd_epi32
            // and _mm_cvttpd_epi32 functions.
            // Converts packed f32/f64 to packed i32.
            "cvtps2dq" | "cvttps2dq" | "cvtpd2dq" | "cvttpd2dq" => {
                let [op] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (op_len, _) = op.layout.ty.simd_size_and_type(*this.tcx);
                let (dest_len, _) = dest.layout.ty.simd_size_and_type(*this.tcx);
                match unprefixed_name {
                    "cvtps2dq" | "cvttps2dq" => {
                        // f32x4 to i32x4 conversion
                        assert_eq!(op_len, 4);
                        assert_eq!(dest_len, op_len);
                    }
                    "cvtpd2dq" | "cvttpd2dq" => {
                        // f64x2 to i32x4 conversion
                        // the last two values are filled with zeros
                        assert_eq!(op_len, 2);
                        assert_eq!(dest_len, 4);
                    }
                    _ => unreachable!(),
                }

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtps2dq
                    // https://www.felixcloutier.com/x86/cvtpd2dq
                    "cvtps2dq" | "cvtpd2dq" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttps2dq
                    // https://www.felixcloutier.com/x86/cvttpd2dq
                    "cvttps2dq" | "cvttpd2dq" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                convert_float_to_int(this, op, rnd, dest)?;
            }
            // Used to implement the _mm_packs_epi16 function.
            // Converts two 16-bit integer vectors to a single 8-bit integer
            // vector with signed saturation.
            "packsswb.128" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                packsswb(this, left, right, dest)?;
            }
            // Used to implement the _mm_packus_epi16 function.
            // Converts two 16-bit signed integer vectors to a single 8-bit
            // unsigned integer vector with saturation.
            "packuswb.128" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                packuswb(this, left, right, dest)?;
            }
            // Used to implement the _mm_packs_epi32 function.
            // Converts two 32-bit integer vectors to a single 16-bit integer
            // vector with signed saturation.
            "packssdw.128" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                packssdw(this, left, right, dest)?;
            }
            // Used to implement _mm_min_sd and _mm_max_sd functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.sd" | "max.sd" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "min.sd" => FloatBinOp::Min,
                    "max.sd" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_simd_float_first::<Double>(this, which, left, right, dest)?;
            }
            // Used to implement _mm_min_pd and _mm_max_pd functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.pd" | "max.pd" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "min.pd" => FloatBinOp::Min,
                    "max.pd" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_simd_float_all::<Double>(this, which, left, right, dest)?;
            }
            // Used to implement the _mm_cmp*_sd functions.
            // Performs a comparison operation on the first component of `left`
            // and `right`, returning 0 if false or `u64::MAX` if true. The remaining
            // components are copied from `left`.
            // _mm_cmp_sd is actually an AVX function where the operation is specified
            // by a const parameter.
            // _mm_cmp{eq,lt,le,gt,ge,neq,nlt,nle,ngt,nge,ord,unord}_sd are SSE2 functions
            // with hard-coded operations.
            "cmp.sd" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which =
                    FloatBinOp::cmp_from_imm(this, this.read_scalar(imm)?.to_i8()?, link_name)?;

                bin_op_simd_float_first::<Double>(this, which, left, right, dest)?;
            }
            // Used to implement the _mm_cmp*_pd functions.
            // Performs a comparison operation on each component of `left`
            // and `right`. For each component, returns 0 if false or `u64::MAX`
            // if true.
            // _mm_cmp_pd is actually an AVX function where the operation is specified
            // by a const parameter.
            // _mm_cmp{eq,lt,le,gt,ge,neq,nlt,nle,ngt,nge,ord,unord}_pd are SSE2 functions
            // with hard-coded operations.
            "cmp.pd" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which =
                    FloatBinOp::cmp_from_imm(this, this.read_scalar(imm)?.to_i8()?, link_name)?;

                bin_op_simd_float_all::<Double>(this, which, left, right, dest)?;
            }
            // Used to implement _mm_{,u}comi{eq,lt,le,gt,ge,neq}_sd functions.
            // Compares the first component of `left` and `right` and returns
            // a scalar value (0 or 1).
            "comieq.sd" | "comilt.sd" | "comile.sd" | "comigt.sd" | "comige.sd" | "comineq.sd"
            | "ucomieq.sd" | "ucomilt.sd" | "ucomile.sd" | "ucomigt.sd" | "ucomige.sd"
            | "ucomineq.sd" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;

                assert_eq!(left_len, right_len);

                let left = this.read_scalar(&this.project_index(&left, 0)?)?.to_f64()?;
                let right = this.read_scalar(&this.project_index(&right, 0)?)?.to_f64()?;
                // The difference between the com* and ucom* variants is signaling
                // of exceptions when either argument is a quiet NaN. We do not
                // support accessing the SSE status register from miri (or from Rust,
                // for that matter), so we treat both variants equally.
                let res = match unprefixed_name {
                    "comieq.sd" | "ucomieq.sd" => left == right,
                    "comilt.sd" | "ucomilt.sd" => left < right,
                    "comile.sd" | "ucomile.sd" => left <= right,
                    "comigt.sd" | "ucomigt.sd" => left > right,
                    "comige.sd" | "ucomige.sd" => left >= right,
                    "comineq.sd" | "ucomineq.sd" => left != right,
                    _ => unreachable!(),
                };
                this.write_scalar(Scalar::from_i32(i32::from(res)), dest)?;
            }
            // Use to implement the _mm_cvtsd_si32, _mm_cvttsd_si32,
            // _mm_cvtsd_si64 and _mm_cvttsd_si64 functions.
            // Converts the first component of `op` from f64 to i32/i64.
            "cvtsd2si" | "cvttsd2si" | "cvtsd2si64" | "cvttsd2si64" => {
                let [op] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let (op, _) = this.project_to_simd(op)?;

                let op = this.read_immediate(&this.project_index(&op, 0)?)?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtsd2si
                    "cvtsd2si" | "cvtsd2si64" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttsd2si
                    "cvttsd2si" | "cvttsd2si64" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let res = this.float_to_int_checked(&op, dest.layout, rnd)?.unwrap_or_else(|| {
                    // Fallback to minimum according to SSE semantics.
                    ImmTy::from_int(dest.layout.size.signed_int_min(), dest.layout)
                });

                this.write_immediate(*res, dest)?;
            }
            // Used to implement the _mm_cvtsd_ss and _mm_cvtss_sd functions.
            // Converts the first f64/f32 from `right` to f32/f64 and copies
            // the remaining elements from `left`
            "cvtsd2ss" | "cvtss2sd" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, _) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                // Convert first element of `right`
                let right0 = this.read_immediate(&this.project_index(&right, 0)?)?;
                let dest0 = this.project_index(&dest, 0)?;
                // `float_to_float_or_int` here will convert from f64 to f32 (cvtsd2ss) or
                // from f32 to f64 (cvtss2sd).
                let res0 = this.float_to_float_or_int(&right0, dest0.layout)?;
                this.write_immediate(*res0, &dest0)?;

                // Copy remaining from `left`
                for i in 1..dest_len {
                    this.copy_op(&this.project_index(&left, i)?, &this.project_index(&dest, i)?)?;
                }
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
