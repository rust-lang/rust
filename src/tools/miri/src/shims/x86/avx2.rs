use rustc_abi::CanonAbi;
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{
    ShiftOp, horizontal_bin_op, int_abs, mask_load, mask_store, mpsadbw, packssdw, packsswb,
    packusdw, packuswb, pmulhrsw, psign, shift_simd_by_scalar, shift_simd_by_simd,
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
            // Used to implement the _mm256_abs_epi{8,16,32} functions.
            // Calculates the absolute value of packed 8/16/32-bit integers.
            "pabs.b" | "pabs.w" | "pabs.d" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                int_abs(this, op, dest)?;
            }
            // Used to implement the _mm256_h{add,adds,sub}_epi{16,32} functions.
            // Horizontally add / add with saturation / subtract adjacent 16/32-bit
            // integer values in `left` and `right`.
            "phadd.w" | "phadd.sw" | "phadd.d" | "phsub.w" | "phsub.sw" | "phsub.d" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (which, saturating) = match unprefixed_name {
                    "phadd.w" | "phadd.d" => (mir::BinOp::Add, false),
                    "phadd.sw" => (mir::BinOp::Add, true),
                    "phsub.w" | "phsub.d" => (mir::BinOp::Sub, false),
                    "phsub.sw" => (mir::BinOp::Sub, true),
                    _ => unreachable!(),
                };

                horizontal_bin_op(this, which, saturating, left, right, dest)?;
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
            // Used to implement the _mm256_madd_epi16 function.
            // Multiplies packed signed 16-bit integers in `left` and `right`, producing
            // intermediate signed 32-bit integers. Horizontally add adjacent pairs of
            // intermediate 32-bit integers, and pack the results in `dest`.
            "pmadd.wd" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

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
            // Used to implement the _mm256_maddubs_epi16 function.
            // Multiplies packed 8-bit unsigned integers from `left` and packed
            // signed 8-bit integers from `right` into 16-bit signed integers. Then,
            // the saturating sum of the products with indices `2*i` and `2*i+1`
            // produces the output at index `i`.
            "pmadd.ub.sw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(dest_len.strict_mul(2), left_len);

                for i in 0..dest_len {
                    let j1 = i.strict_mul(2);
                    let left1 = this.read_scalar(&this.project_index(&left, j1)?)?.to_u8()?;
                    let right1 = this.read_scalar(&this.project_index(&right, j1)?)?.to_i8()?;

                    let j2 = j1.strict_add(1);
                    let left2 = this.read_scalar(&this.project_index(&left, j2)?)?.to_u8()?;
                    let right2 = this.read_scalar(&this.project_index(&right, j2)?)?.to_i8()?;

                    let dest = this.project_index(&dest, i)?;

                    // Multiplication of a u8 and an i8 into an i16 cannot overflow.
                    let mul1 = i16::from(left1).strict_mul(right1.into());
                    let mul2 = i16::from(left2).strict_mul(right2.into());
                    let res = mul1.saturating_add(mul2);

                    this.write_scalar(Scalar::from_i16(res), &dest)?;
                }
            }
            // Used to implement the _mm_maskload_epi32, _mm_maskload_epi64,
            // _mm256_maskload_epi32 and _mm256_maskload_epi64 functions.
            // For the element `i`, if the high bit of the `i`-th element of `mask`
            // is one, it is loaded from `ptr.wrapping_add(i)`, otherwise zero is
            // loaded.
            "maskload.d" | "maskload.q" | "maskload.d.256" | "maskload.q.256" => {
                let [ptr, mask] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                mask_load(this, ptr, mask, dest)?;
            }
            // Used to implement the _mm_maskstore_epi32, _mm_maskstore_epi64,
            // _mm256_maskstore_epi32 and _mm256_maskstore_epi64 functions.
            // For the element `i`, if the high bit of the element `i`-th of `mask`
            // is one, it is stored into `ptr.wapping_add(i)`.
            // Unlike SSE2's _mm_maskmoveu_si128, these are not non-temporal stores.
            "maskstore.d" | "maskstore.q" | "maskstore.d.256" | "maskstore.q.256" => {
                let [ptr, mask, value] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                mask_store(this, ptr, mask, value)?;
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
            // Used to implement the _mm256_permutevar8x32_epi32 and
            // _mm256_permutevar8x32_ps function.
            // Shuffles `left` using the three low bits of each element of `right`
            // as indices.
            "permd" | "permps" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let dest = this.project_index(&dest, i)?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_u32()?;
                    let left = this.project_index(&left, (right & 0b111).into())?;

                    this.copy_op(&left, &dest)?;
                }
            }
            // Used to implement the _mm256_permute2x128_si256 function.
            // Shuffles 128-bit blocks of `a` and `b` using `imm` as pattern.
            "vperm2i128" => {
                let [left, right, imm] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                assert_eq!(left.layout.size.bits(), 256);
                assert_eq!(right.layout.size.bits(), 256);
                assert_eq!(dest.layout.size.bits(), 256);

                // Transmute to `[i128; 2]`

                let array_layout =
                    this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.i128, 2))?;
                let left = left.transmute(array_layout, this)?;
                let right = right.transmute(array_layout, this)?;
                let dest = dest.transmute(array_layout, this)?;

                let imm = this.read_scalar(imm)?.to_u8()?;

                for i in 0..2 {
                    let dest = this.project_index(&dest, i)?;
                    let src = match (imm >> i.strict_mul(4)) & 0b11 {
                        0 => this.project_index(&left, 0)?,
                        1 => this.project_index(&left, 1)?,
                        2 => this.project_index(&right, 0)?,
                        3 => this.project_index(&right, 1)?,
                        _ => unreachable!(),
                    };

                    this.copy_op(&src, &dest)?;
                }
            }
            // Used to implement the _mm256_sad_epu8 function.
            // Compute the absolute differences of packed unsigned 8-bit integers
            // in `left` and `right`, then horizontally sum each consecutive 8
            // differences to produce four unsigned 16-bit integers, and pack
            // these unsigned 16-bit integers in the low 16 bits of 64-bit elements
            // in `dest`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_sad_epu8
            "psad.bw" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(left_len, dest_len.strict_mul(8));

                for i in 0..dest_len {
                    let dest = this.project_index(&dest, i)?;

                    let mut acc: u16 = 0;
                    for j in 0..8 {
                        let src_index = i.strict_mul(8).strict_add(j);

                        let left = this.project_index(&left, src_index)?;
                        let left = this.read_scalar(&left)?.to_u8()?;

                        let right = this.project_index(&right, src_index)?;
                        let right = this.read_scalar(&right)?.to_u8()?;

                        acc = acc.strict_add(left.abs_diff(right).into());
                    }

                    this.write_scalar(Scalar::from_u64(acc.into()), &dest)?;
                }
            }
            // Used to implement the _mm256_shuffle_epi8 intrinsic.
            // Shuffles bytes from `left` using `right` as pattern.
            // Each 128-bit block is shuffled independently.
            "pshuf.b" => {
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
                        // Shuffle each 128-bit (16-byte) block independently.
                        let j = u64::from(right % 16).strict_add(i & !15);
                        this.read_scalar(&this.project_index(&left, j)?)?
                    } else {
                        // If the highest bit in `right` is 1, write zero.
                        Scalar::from_u8(0)
                    };

                    this.write_scalar(res, &dest)?;
                }
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
            // Used to implement the _mm{,256}_{sllv,srlv,srav}_epi{32,64} functions
            // (except _mm{,256}_srav_epi64, which are not available in AVX2).
            "psllv.d" | "psllv.d.256" | "psllv.q" | "psllv.q.256" | "psrlv.d" | "psrlv.d.256"
            | "psrlv.q" | "psrlv.q.256" | "psrav.d" | "psrav.d.256" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "psllv.d" | "psllv.d.256" | "psllv.q" | "psllv.q.256" => ShiftOp::Left,
                    "psrlv.d" | "psrlv.d.256" | "psrlv.q" | "psrlv.q.256" => ShiftOp::RightLogic,
                    "psrav.d" | "psrav.d.256" => ShiftOp::RightArith,
                    _ => unreachable!(),
                };

                shift_simd_by_simd(this, left, right, which, dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
