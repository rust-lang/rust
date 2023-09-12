use rustc_apfloat::{
    ieee::{Double, Single},
    Float as _, FloatConvert as _,
};
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use super::FloatCmpOp;
use crate::*;
use shims::foreign_items::EmulateByNameResult;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_x86_sse2_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();
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
            // Used to implement the _mm_avg_epu8 function.
            // Averages packed unsigned 8-bit integers in `left` and `right`.
            "pavg.b" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u8()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_u8()?;
                    let dest = this.project_index(&dest, i)?;

                    // Values are expanded from u8 to u16, so adds cannot overflow.
                    let res = u16::from(left)
                        .checked_add(u16::from(right))
                        .unwrap()
                        .checked_add(1)
                        .unwrap()
                        / 2;
                    this.write_scalar(Scalar::from_u8(res.try_into().unwrap()), &dest)?;
                }
            }
            // Used to implement the _mm_avg_epu16 function.
            // Averages packed unsigned 16-bit integers in `left` and `right`.
            "pavg.w" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u16()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_u16()?;
                    let dest = this.project_index(&dest, i)?;

                    // Values are expanded from u16 to u32, so adds cannot overflow.
                    let res = u32::from(left)
                        .checked_add(u32::from(right))
                        .unwrap()
                        .checked_add(1)
                        .unwrap()
                        / 2;
                    this.write_scalar(Scalar::from_u16(res.try_into().unwrap()), &dest)?;
                }
            }
            // Used to implement the _mm_mulhi_epi16 function.
            "pmulh.w" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_i16()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_i16()?;
                    let dest = this.project_index(&dest, i)?;

                    // Values are expanded from i16 to i32, so multiplication cannot overflow.
                    let res = i32::from(left).checked_mul(i32::from(right)).unwrap() >> 16;
                    this.write_scalar(Scalar::from_int(res, Size::from_bits(16)), &dest)?;
                }
            }
            // Used to implement the _mm_mulhi_epu16 function.
            "pmulhu.w" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u16()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_u16()?;
                    let dest = this.project_index(&dest, i)?;

                    // Values are expanded from u16 to u32, so multiplication cannot overflow.
                    let res = u32::from(left).checked_mul(u32::from(right)).unwrap() >> 16;
                    this.write_scalar(Scalar::from_u16(res.try_into().unwrap()), &dest)?;
                }
            }
            // Used to implement the _mm_mul_epu32 function.
            // Multiplies the the low unsigned 32-bit integers from each packed
            // 64-bit element and stores the result as 64-bit unsigned integers.
            "pmulu.dq" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // left and right are u32x4, dest is u64x2
                assert_eq!(left_len, 4);
                assert_eq!(right_len, 4);
                assert_eq!(dest_len, 2);

                for i in 0..dest_len {
                    let op_i = i.checked_mul(2).unwrap();
                    let left = this.read_scalar(&this.project_index(&left, op_i)?)?.to_u32()?;
                    let right = this.read_scalar(&this.project_index(&right, op_i)?)?.to_u32()?;
                    let dest = this.project_index(&dest, i)?;

                    // The multiplication will not overflow because stripping the
                    // operands are expanded from 32-bit to 64-bit.
                    let res = u64::from(left).checked_mul(u64::from(right)).unwrap();
                    this.write_scalar(Scalar::from_u64(res), &dest)?;
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
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // left and right are u8x16, dest is u64x2
                assert_eq!(left_len, right_len);
                assert_eq!(left_len, 16);
                assert_eq!(dest_len, 2);

                for i in 0..dest_len {
                    let dest = this.project_index(&dest, i)?;

                    let mut res: u16 = 0;
                    let n = left_len.checked_div(dest_len).unwrap();
                    for j in 0..n {
                        let op_i = j.checked_add(i.checked_mul(n).unwrap()).unwrap();
                        let left = this.read_scalar(&this.project_index(&left, op_i)?)?.to_u8()?;
                        let right =
                            this.read_scalar(&this.project_index(&right, op_i)?)?.to_u8()?;

                        res = res.checked_add(left.abs_diff(right).into()).unwrap();
                    }

                    this.write_scalar(Scalar::from_u64(res.into()), &dest)?;
                }
            }
            // Used to implement the _mm_{sll,srl,sra}_epi16 functions.
            // Shifts 16-bit packed integers in left by the amount in right.
            // Both operands are vectors of 16-bit integers. However, right is
            // interpreted as a single 64-bit integer (remaining bits are ignored).
            // For logic shifts, when right is larger than 15, zero is produced.
            // For arithmetic shifts, when right is larger than 15, the sign bit
            // is copied to remaining bits.
            "psll.w" | "psrl.w" | "psra.w" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                enum ShiftOp {
                    Sll,
                    Srl,
                    Sra,
                }
                let which = match unprefixed_name {
                    "psll.w" => ShiftOp::Sll,
                    "psrl.w" => ShiftOp::Srl,
                    "psra.w" => ShiftOp::Sra,
                    _ => unreachable!(),
                };

                // Get the 64-bit shift operand and convert it to the type expected
                // by checked_{shl,shr} (u32).
                // It is ok to saturate the value to u32::MAX because any value
                // above 15 will produce the same result.
                let shift = extract_first_u64(this, &right)?.try_into().unwrap_or(u32::MAX);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u16()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = match which {
                        ShiftOp::Sll => left.checked_shl(shift).unwrap_or(0),
                        ShiftOp::Srl => left.checked_shr(shift).unwrap_or(0),
                        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                        ShiftOp::Sra => {
                            // Convert u16 to i16 to use arithmetic shift
                            let left = left as i16;
                            // Copy the sign bit to the remaining bits
                            left.checked_shr(shift).unwrap_or(left >> 15) as u16
                        }
                    };

                    this.write_scalar(Scalar::from_u16(res), &dest)?;
                }
            }
            // Used to implement the _mm_{sll,srl,sra}_epi32 functions.
            // 32-bit equivalent to the shift functions above.
            "psll.d" | "psrl.d" | "psra.d" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                enum ShiftOp {
                    Sll,
                    Srl,
                    Sra,
                }
                let which = match unprefixed_name {
                    "psll.d" => ShiftOp::Sll,
                    "psrl.d" => ShiftOp::Srl,
                    "psra.d" => ShiftOp::Sra,
                    _ => unreachable!(),
                };

                // Get the 64-bit shift operand and convert it to the type expected
                // by checked_{shl,shr} (u32).
                // It is ok to saturate the value to u32::MAX because any value
                // above 31 will produce the same result.
                let shift = extract_first_u64(this, &right)?.try_into().unwrap_or(u32::MAX);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u32()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = match which {
                        ShiftOp::Sll => left.checked_shl(shift).unwrap_or(0),
                        ShiftOp::Srl => left.checked_shr(shift).unwrap_or(0),
                        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                        ShiftOp::Sra => {
                            // Convert u32 to i32 to use arithmetic shift
                            let left = left as i32;
                            // Copy the sign bit to the remaining bits
                            left.checked_shr(shift).unwrap_or(left >> 31) as u32
                        }
                    };

                    this.write_scalar(Scalar::from_u32(res), &dest)?;
                }
            }
            // Used to implement the _mm_{sll,srl}_epi64 functions.
            // 64-bit equivalent to the shift functions above, except _mm_sra_epi64,
            // which is not available in SSE2.
            "psll.q" | "psrl.q" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                enum ShiftOp {
                    Sll,
                    Srl,
                }
                let which = match unprefixed_name {
                    "psll.q" => ShiftOp::Sll,
                    "psrl.q" => ShiftOp::Srl,
                    _ => unreachable!(),
                };

                // Get the 64-bit shift operand and convert it to the type expected
                // by checked_{shl,shr} (u32).
                // It is ok to saturate the value to u32::MAX because any value
                // above 63 will produce the same result.
                let shift = this
                    .read_scalar(&this.project_index(&right, 0)?)?
                    .to_u64()?
                    .try_into()
                    .unwrap_or(u32::MAX);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u64()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = match which {
                        ShiftOp::Sll => left.checked_shl(shift).unwrap_or(0),
                        ShiftOp::Srl => left.checked_shr(shift).unwrap_or(0),
                    };

                    this.write_scalar(Scalar::from_u64(res), &dest)?;
                }
            }
            // Used to implement the _mm_cvtepi32_ps function.
            // Converts packed i32 to packed f32.
            // FIXME: Can we get rid of this intrinsic and just use simd_as?
            "cvtdq2ps" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_i32()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = Scalar::from_f32(Single::from_i128(op.into()).value);
                    this.write_scalar(res, &dest)?;
                }
            }
            // Used to implement the _mm_cvtps_epi32 and _mm_cvttps_epi32 functions.
            // Converts packed f32 to packed i32.
            "cvtps2dq" | "cvttps2dq" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtps2dq
                    "cvtps2dq" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttps2dq
                    "cvttps2dq" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_f32()?;
                    let dest = this.project_index(&dest, i)?;

                    let res =
                        this.float_to_int_checked(op, dest.layout.ty, rnd).unwrap_or_else(|| {
                            // Fallback to minimum acording to SSE2 semantics.
                            Scalar::from_i32(i32::MIN)
                        });
                    this.write_scalar(res, &dest)?;
                }
            }
            // Used to implement the _mm_packs_epi16 function.
            // Converts two 16-bit integer vectors to a single 8-bit integer
            // vector with signed saturation.
            "packsswb.128" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // left and right are i16x8, dest is i8x16
                assert_eq!(left_len, 8);
                assert_eq!(right_len, 8);
                assert_eq!(dest_len, 16);

                for i in 0..left_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_i16()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_i16()?;
                    let left_dest = this.project_index(&dest, i)?;
                    let right_dest = this.project_index(&dest, i.checked_add(left_len).unwrap())?;

                    let left_res =
                        i8::try_from(left).unwrap_or(if left < 0 { i8::MIN } else { i8::MAX });
                    let right_res =
                        i8::try_from(right).unwrap_or(if right < 0 { i8::MIN } else { i8::MAX });

                    this.write_scalar(Scalar::from_int(left_res, Size::from_bits(8)), &left_dest)?;
                    this.write_scalar(
                        Scalar::from_int(right_res, Size::from_bits(8)),
                        &right_dest,
                    )?;
                }
            }
            // Used to implement the _mm_packus_epi16 function.
            // Converts two 16-bit signed integer vectors to a single 8-bit
            // unsigned integer vector with saturation.
            "packuswb.128" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // left and right are i16x8, dest is u8x16
                assert_eq!(left_len, 8);
                assert_eq!(right_len, 8);
                assert_eq!(dest_len, 16);

                for i in 0..left_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_i16()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_i16()?;
                    let left_dest = this.project_index(&dest, i)?;
                    let right_dest = this.project_index(&dest, i.checked_add(left_len).unwrap())?;

                    let left_res = u8::try_from(left).unwrap_or(if left < 0 { 0 } else { u8::MAX });
                    let right_res =
                        u8::try_from(right).unwrap_or(if right < 0 { 0 } else { u8::MAX });

                    this.write_scalar(Scalar::from_u8(left_res), &left_dest)?;
                    this.write_scalar(Scalar::from_u8(right_res), &right_dest)?;
                }
            }
            // Used to implement the _mm_packs_epi32 function.
            // Converts two 16-bit integer vectors to a single 8-bit integer
            // vector with signed saturation.
            "packssdw.128" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // left and right are i32x4, dest is i16x8
                assert_eq!(left_len, 4);
                assert_eq!(right_len, 4);
                assert_eq!(dest_len, 8);

                for i in 0..left_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_i32()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_i32()?;
                    let left_dest = this.project_index(&dest, i)?;
                    let right_dest = this.project_index(&dest, i.checked_add(left_len).unwrap())?;

                    let left_res =
                        i16::try_from(left).unwrap_or(if left < 0 { i16::MIN } else { i16::MAX });
                    let right_res =
                        i16::try_from(right).unwrap_or(if right < 0 { i16::MIN } else { i16::MAX });

                    this.write_scalar(Scalar::from_int(left_res, Size::from_bits(16)), &left_dest)?;
                    this.write_scalar(
                        Scalar::from_int(right_res, Size::from_bits(16)),
                        &right_dest,
                    )?;
                }
            }
            // Used to implement _mm_min_sd and _mm_max_sd functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.sd" | "max.sd" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "min.sd" => FloatBinOp::Min,
                    "max.sd" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_sd(this, which, left, right, dest)?;
            }
            // Used to implement _mm_min_pd and _mm_max_pd functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.pd" | "max.pd" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "min.pd" => FloatBinOp::Min,
                    "max.pd" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_pd(this, which, left, right, dest)?;
            }
            // Used to implement _mm_sqrt_sd functions.
            // Performs the operations on the first component of `op` and
            // copies the remaining components from `op`.
            "sqrt.sd" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                let op0 = this.read_scalar(&this.project_index(&op, 0)?)?.to_u64()?;
                // FIXME using host floats
                let res0 = Scalar::from_u64(f64::from_bits(op0).sqrt().to_bits());
                this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

                for i in 1..dest_len {
                    this.copy_op(
                        &this.project_index(&op, i)?,
                        &this.project_index(&dest, i)?,
                        /*allow_transmute*/ false,
                    )?;
                }
            }
            // Used to implement _mm_sqrt_pd functions.
            // Performs the operations on all components of `op`.
            "sqrt.pd" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_u64()?;
                    let dest = this.project_index(&dest, i)?;

                    // FIXME using host floats
                    let res = Scalar::from_u64(f64::from_bits(op).sqrt().to_bits());

                    this.write_scalar(res, &dest)?;
                }
            }
            // Used to implement the _mm_cmp*_sd function.
            // Performs a comparison operation on the first component of `left`
            // and `right`, returning 0 if false or `u64::MAX` if true. The remaining
            // components are copied from `left`.
            "cmp.sd" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = FloatBinOp::Cmp(FloatCmpOp::from_intrinsic_imm(
                    this.read_scalar(imm)?.to_i8()?,
                    "llvm.x86.sse2.cmp.sd",
                )?);

                bin_op_sd(this, which, left, right, dest)?;
            }
            // Used to implement the _mm_cmp*_pd functions.
            // Performs a comparison operation on each component of `left`
            // and `right`. For each component, returns 0 if false or `u64::MAX`
            // if true.
            "cmp.pd" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = FloatBinOp::Cmp(FloatCmpOp::from_intrinsic_imm(
                    this.read_scalar(imm)?.to_i8()?,
                    "llvm.x86.sse2.cmp.pd",
                )?);

                bin_op_pd(this, which, left, right, dest)?;
            }
            // Used to implement _mm_{,u}comi{eq,lt,le,gt,ge,neq}_sd functions.
            // Compares the first component of `left` and `right` and returns
            // a scalar value (0 or 1).
            "comieq.sd" | "comilt.sd" | "comile.sd" | "comigt.sd" | "comige.sd" | "comineq.sd"
            | "ucomieq.sd" | "ucomilt.sd" | "ucomile.sd" | "ucomigt.sd" | "ucomige.sd"
            | "ucomineq.sd" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;

                assert_eq!(left_len, right_len);

                let left = this.read_scalar(&this.project_index(&left, 0)?)?.to_f64()?;
                let right = this.read_scalar(&this.project_index(&right, 0)?)?.to_f64()?;
                // The difference between the com* and *ucom variants is signaling
                // of exceptions when either argument is a quiet NaN. We do not
                // support accessing the SSE status register from miri (or from Rust,
                // for that matter), so we treat equally both variants.
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
            // Used to implement the _mm_cvtpd_ps function.
            // Converts packed f32 to packed f64.
            "cvtpd2ps" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // op is f64x2, dest is f32x4
                assert_eq!(op_len, 2);
                assert_eq!(dest_len, 4);

                for i in 0..op_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_f64()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = op.convert(/*loses_info*/ &mut false).value;
                    this.write_scalar(Scalar::from_f32(res), &dest)?;
                }
                // Fill the remaining with zeros
                for i in op_len..dest_len {
                    let dest = this.project_index(&dest, i)?;
                    this.write_scalar(Scalar::from_u32(0), &dest)?;
                }
            }
            // Used to implement the _mm_cvtps_pd function.
            // Converts packed f64 to packed f32.
            "cvtps2pd" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // op is f32x4, dest is f64x2
                assert_eq!(op_len, 4);
                assert_eq!(dest_len, 2);

                for i in 0..dest_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_f32()?;
                    let dest = this.project_index(&dest, i)?;

                    let res = op.convert(/*loses_info*/ &mut false).value;
                    this.write_scalar(Scalar::from_f64(res), &dest)?;
                }
                // the two remaining f32 are ignored
            }
            // Used to implement the _mm_cvtpd_epi32 and _mm_cvttpd_epi32 functions.
            // Converts packed f64 to packed i32.
            "cvtpd2dq" | "cvttpd2dq" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // op is f64x2, dest is i32x4
                assert_eq!(op_len, 2);
                assert_eq!(dest_len, 4);

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtpd2dq
                    "cvtpd2dq" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttpd2dq
                    "cvttpd2dq" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                for i in 0..op_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_f64()?;
                    let dest = this.project_index(&dest, i)?;

                    let res =
                        this.float_to_int_checked(op, dest.layout.ty, rnd).unwrap_or_else(|| {
                            // Fallback to minimum acording to SSE2 semantics.
                            Scalar::from_i32(i32::MIN)
                        });
                    this.write_scalar(res, &dest)?;
                }
                // Fill the remaining with zeros
                for i in op_len..dest_len {
                    let dest = this.project_index(&dest, i)?;
                    this.write_scalar(Scalar::from_i32(0), &dest)?;
                }
            }
            // Use to implement the _mm_cvtsd_si32 and _mm_cvttsd_si32 functions.
            // Converts the first component of `op` from f64 to i32.
            "cvtsd2si" | "cvttsd2si" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, _) = this.operand_to_simd(op)?;

                let op = this.read_scalar(&this.project_index(&op, 0)?)?.to_f64()?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtsd2si
                    "cvtsd2si" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttsd2si
                    "cvttsd2si" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let res = this.float_to_int_checked(op, dest.layout.ty, rnd).unwrap_or_else(|| {
                    // Fallback to minimum acording to SSE semantics.
                    Scalar::from_i32(i32::MIN)
                });

                this.write_scalar(res, dest)?;
            }
            // Use to implement the _mm_cvtsd_si64 and _mm_cvttsd_si64 functions.
            // Converts the first component of `op` from f64 to i64.
            "cvtsd2si64" | "cvttsd2si64" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, _) = this.operand_to_simd(op)?;

                let op = this.read_scalar(&this.project_index(&op, 0)?)?.to_f64()?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtsd2si
                    "cvtsd2si64" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttsd2si
                    "cvttsd2si64" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let res = this.float_to_int_checked(op, dest.layout.ty, rnd).unwrap_or_else(|| {
                    // Fallback to minimum acording to SSE semantics.
                    Scalar::from_i64(i64::MIN)
                });

                this.write_scalar(res, dest)?;
            }
            // Used to implement the _mm_cvtsd_ss and _mm_cvtss_sd functions.
            // Converts the first f64/f32 from `right` to f32/f64 and copies
            // the remaining elements from `left`
            "cvtsd2ss" | "cvtss2sd" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, _) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                // Convert first element of `right`
                let right0 = this.read_immediate(&this.project_index(&right, 0)?)?;
                let dest0 = this.project_index(&dest, 0)?;
                // `float_to_float_or_int` here will convert from f64 to f32 (cvtsd2ss) or
                // from f32 to f64 (cvtss2sd).
                let res0 = this.float_to_float_or_int(&right0, dest0.layout.ty)?;
                this.write_immediate(res0, &dest0)?;

                // Copy remianing from `left`
                for i in 1..dest_len {
                    this.copy_op(
                        &this.project_index(&left, i)?,
                        &this.project_index(&dest, i)?,
                        /*allow_transmute*/ false,
                    )?;
                }
            }
            // Used to implement the _mm_movemask_pd function.
            // Returns a scalar integer where the i-th bit is the highest
            // bit of the i-th component of `op`.
            // https://www.felixcloutier.com/x86/movmskpd
            "movmsk.pd" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, op_len) = this.operand_to_simd(op)?;

                let mut res = 0;
                for i in 0..op_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?;
                    let op = op.to_u64()?;

                    // Extract the highest bit of `op` and place it in the `i`-th bit of `res`
                    res |= (op >> 63) << i;
                }

                this.write_scalar(Scalar::from_u32(res.try_into().unwrap()), dest)?;
            }
            _ => return Ok(EmulateByNameResult::NotSupported),
        }
        Ok(EmulateByNameResult::NeedsJumping)
    }
}

/// Takes a 128-bit vector, transmutes it to `[u64; 2]` and extracts
/// the first value.
fn extract_first_u64<'tcx>(
    this: &crate::MiriInterpCx<'_, 'tcx>,
    op: &MPlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, u64> {
    // Transmute vector to `[u64; 2]`
    let u64_array_layout = this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.u64, 2))?;
    let op = op.transmute(u64_array_layout, this)?;

    // Get the first u64 from the array
    this.read_scalar(&this.project_index(&op, 0)?)?.to_u64()
}

#[derive(Copy, Clone)]
enum FloatBinOp {
    /// Comparison
    Cmp(FloatCmpOp),
    /// Minimum value (with SSE semantics)
    ///
    /// <https://www.felixcloutier.com/x86/minsd>
    /// <https://www.felixcloutier.com/x86/minpd>
    Min,
    /// Maximum value (with SSE semantics)
    ///
    /// <https://www.felixcloutier.com/x86/maxsd>
    /// <https://www.felixcloutier.com/x86/maxpd>
    Max,
}

/// Performs `which` scalar operation on `left` and `right` and returns
/// the result.
// FIXME make this generic over apfloat type to reduce code duplicaton with bin_op_f32
fn bin_op_f64<'tcx>(
    which: FloatBinOp,
    left: &ImmTy<'tcx, Provenance>,
    right: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    match which {
        FloatBinOp::Cmp(which) => {
            let left = left.to_scalar().to_f64()?;
            let right = right.to_scalar().to_f64()?;
            // FIXME: Make sure that these operations match the semantics of cmppd
            let res = match which {
                FloatCmpOp::Eq => left == right,
                FloatCmpOp::Lt => left < right,
                FloatCmpOp::Le => left <= right,
                FloatCmpOp::Unord => left.is_nan() || right.is_nan(),
                FloatCmpOp::Neq => left != right,
                FloatCmpOp::Nlt => !(left < right),
                FloatCmpOp::Nle => !(left <= right),
                FloatCmpOp::Ord => !left.is_nan() && !right.is_nan(),
            };
            Ok(Scalar::from_u64(if res { u64::MAX } else { 0 }))
        }
        FloatBinOp::Min => {
            let left = left.to_scalar().to_f64()?;
            let right = right.to_scalar().to_f64()?;
            // SSE semantics to handle zero and NaN. Note that `x == Single::ZERO`
            // is true when `x` is either +0 or -0.
            if (left == Double::ZERO && right == Double::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left >= right
            {
                Ok(Scalar::from_f64(right))
            } else {
                Ok(Scalar::from_f64(left))
            }
        }
        FloatBinOp::Max => {
            let left = left.to_scalar().to_f64()?;
            let right = right.to_scalar().to_f64()?;
            // SSE semantics to handle zero and NaN. Note that `x == Single::ZERO`
            // is true when `x` is either +0 or -0.
            if (left == Double::ZERO && right == Double::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left <= right
            {
                Ok(Scalar::from_f64(right))
            } else {
                Ok(Scalar::from_f64(left))
            }
        }
    }
}

/// Performs `which` operation on the first component of `left` and `right`
/// and copies the other components from `left`. The result is stored in `dest`.
fn bin_op_sd<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatBinOp,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    let res0 = bin_op_f64(
        which,
        &this.read_immediate(&this.project_index(&left, 0)?)?,
        &this.read_immediate(&this.project_index(&right, 0)?)?,
    )?;
    this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

    for i in 1..dest_len {
        this.copy_op(
            &this.project_index(&left, i)?,
            &this.project_index(&dest, i)?,
            /*allow_transmute*/ false,
        )?;
    }

    Ok(())
}

/// Performs `which` operation on each component of `left` and
/// `right`, storing the result is stored in `dest`.
fn bin_op_pd<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatBinOp,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    for i in 0..dest_len {
        let left = this.read_immediate(&this.project_index(&left, i)?)?;
        let right = this.read_immediate(&this.project_index(&right, i)?)?;
        let dest = this.project_index(&dest, i)?;

        let res = bin_op_f64(which, &left, &right)?;
        this.write_scalar(res, &dest)?;
    }

    Ok(())
}
