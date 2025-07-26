use rustc_abi::CanonAbi;
use rustc_apfloat::ieee::{Double, Single};
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{
    FloatBinOp, FloatUnaryOp, bin_op_simd_float_all, conditional_dot_product, convert_float_to_int,
    horizontal_bin_op, mask_load, mask_store, round_all, test_bits_masked, test_high_bits_masked,
    unary_op_ps,
};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_avx_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "avx")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.avx.").unwrap();

        match unprefixed_name {
            // Used to implement _mm256_min_ps and _mm256_max_ps functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.ps.256" | "max.ps.256" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "min.ps.256" => FloatBinOp::Min,
                    "max.ps.256" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_simd_float_all::<Single>(this, which, left, right, dest)?;
            }
            // Used to implement _mm256_min_pd and _mm256_max_pd functions.
            "min.pd.256" | "max.pd.256" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "min.pd.256" => FloatBinOp::Min,
                    "max.pd.256" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_simd_float_all::<Double>(this, which, left, right, dest)?;
            }
            // Used to implement the _mm256_round_ps function.
            // Rounds the elements of `op` according to `rounding`.
            "round.ps.256" => {
                let [op, rounding] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                round_all::<rustc_apfloat::ieee::Single>(this, op, rounding, dest)?;
            }
            // Used to implement the _mm256_round_pd function.
            // Rounds the elements of `op` according to `rounding`.
            "round.pd.256" => {
                let [op, rounding] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                round_all::<rustc_apfloat::ieee::Double>(this, op, rounding, dest)?;
            }
            // Used to implement _mm256_{rcp,rsqrt}_ps functions.
            // Performs the operations on all components of `op`.
            "rcp.ps.256" | "rsqrt.ps.256" => {
                let [op] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "rcp.ps.256" => FloatUnaryOp::Rcp,
                    "rsqrt.ps.256" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ps(this, which, op, dest)?;
            }
            // Used to implement the _mm256_dp_ps function.
            "dp.ps.256" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                conditional_dot_product(this, left, right, imm, dest)?;
            }
            // Used to implement the _mm256_h{add,sub}_p{s,d} functions.
            // Horizontally add/subtract adjacent floating point values
            // in `left` and `right`.
            "hadd.ps.256" | "hadd.pd.256" | "hsub.ps.256" | "hsub.pd.256" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "hadd.ps.256" | "hadd.pd.256" => mir::BinOp::Add,
                    "hsub.ps.256" | "hsub.pd.256" => mir::BinOp::Sub,
                    _ => unreachable!(),
                };

                horizontal_bin_op(this, which, /*saturating*/ false, left, right, dest)?;
            }
            // Used to implement the _mm256_cmp_ps function.
            // Performs a comparison operation on each component of `left`
            // and `right`. For each component, returns 0 if false or u32::MAX
            // if true.
            "cmp.ps.256" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which =
                    FloatBinOp::cmp_from_imm(this, this.read_scalar(imm)?.to_i8()?, link_name)?;

                bin_op_simd_float_all::<Single>(this, which, left, right, dest)?;
            }
            // Used to implement the _mm256_cmp_pd function.
            // Performs a comparison operation on each component of `left`
            // and `right`. For each component, returns 0 if false or u64::MAX
            // if true.
            "cmp.pd.256" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let which =
                    FloatBinOp::cmp_from_imm(this, this.read_scalar(imm)?.to_i8()?, link_name)?;

                bin_op_simd_float_all::<Double>(this, which, left, right, dest)?;
            }
            // Used to implement the _mm256_cvtps_epi32, _mm256_cvttps_epi32, _mm256_cvtpd_epi32
            // and _mm256_cvttpd_epi32 functions.
            // Converts packed f32/f64 to packed i32.
            "cvt.ps2dq.256" | "cvtt.ps2dq.256" | "cvt.pd2dq.256" | "cvtt.pd2dq.256" => {
                let [op] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    "cvt.ps2dq.256" | "cvt.pd2dq.256" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    "cvtt.ps2dq.256" | "cvtt.pd2dq.256" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                convert_float_to_int(this, op, rnd, dest)?;
            }
            // Used to implement the _mm_permutevar_ps and _mm256_permutevar_ps functions.
            // Shuffles 32-bit floats from `data` using `control` as control. Each 128-bit
            // chunk is shuffled independently: this means that we view the vector as a
            // sequence of 4-element arrays, and we shuffle each of these arrays, where
            // `control` determines which element of the current `data` array is written.
            "vpermilvar.ps" | "vpermilvar.ps.256" => {
                let [data, control] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (data, data_len) = this.project_to_simd(data)?;
                let (control, control_len) = this.project_to_simd(control)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, data_len);
                assert_eq!(dest_len, control_len);

                for i in 0..dest_len {
                    let control = this.project_index(&control, i)?;

                    // Each 128-bit chunk is shuffled independently. Since each chunk contains
                    // four 32-bit elements, only two bits from `control` are used. To read the
                    // value from the current chunk, add the destination index truncated to a multiple
                    // of 4.
                    let chunk_base = i & !0b11;
                    let src_i = u64::from(this.read_scalar(&control)?.to_u32()? & 0b11)
                        .strict_add(chunk_base);

                    this.copy_op(
                        &this.project_index(&data, src_i)?,
                        &this.project_index(&dest, i)?,
                    )?;
                }
            }
            // Used to implement the _mm_permutevar_pd and _mm256_permutevar_pd functions.
            // Shuffles 64-bit floats from `left` using `right` as control. Each 128-bit
            // chunk is shuffled independently: this means that we view the vector as
            // a sequence of 2-element arrays, and we shuffle each of these arrays,
            // where `right` determines which element of the current `left` array is
            // written.
            "vpermilvar.pd" | "vpermilvar.pd.256" => {
                let [data, control] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (data, data_len) = this.project_to_simd(data)?;
                let (control, control_len) = this.project_to_simd(control)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, data_len);
                assert_eq!(dest_len, control_len);

                for i in 0..dest_len {
                    let control = this.project_index(&control, i)?;

                    // Each 128-bit chunk is shuffled independently. Since each chunk contains
                    // two 64-bit elements, only the second bit from `control` is used (yes, the
                    // second instead of the first, ask Intel). To read the value from the current
                    // chunk, add the destination index truncated to a multiple of 2.
                    let chunk_base = i & !1;
                    let src_i =
                        ((this.read_scalar(&control)?.to_u64()? >> 1) & 1).strict_add(chunk_base);

                    this.copy_op(
                        &this.project_index(&data, src_i)?,
                        &this.project_index(&dest, i)?,
                    )?;
                }
            }
            // Used to implement the _mm256_permute2f128_ps, _mm256_permute2f128_pd and
            // _mm256_permute2f128_si256 functions. Regardless of the suffix in the name
            // thay all can be considered to operate on vectors of 128-bit elements.
            // For each 128-bit element of `dest`, copies one from `left`, `right` or
            // zero, according to `imm`.
            "vperm2f128.ps.256" | "vperm2f128.pd.256" | "vperm2f128.si.256" => {
                let [left, right, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                assert_eq!(dest.layout, left.layout);
                assert_eq!(dest.layout, right.layout);
                assert_eq!(dest.layout.size.bits(), 256);

                // Transmute to `[u128; 2]` to process each 128-bit chunk independently.
                let u128x2_layout =
                    this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.u128, 2))?;
                let left = left.transmute(u128x2_layout, this)?;
                let right = right.transmute(u128x2_layout, this)?;
                let dest = dest.transmute(u128x2_layout, this)?;

                let imm = this.read_scalar(imm)?.to_u8()?;

                for i in 0..2 {
                    let dest = this.project_index(&dest, i)?;

                    let imm = match i {
                        0 => imm & 0xF,
                        1 => imm >> 4,
                        _ => unreachable!(),
                    };
                    if imm & 0b100 != 0 {
                        this.write_scalar(Scalar::from_u128(0), &dest)?;
                    } else {
                        let src = match imm {
                            0b00 => this.project_index(&left, 0)?,
                            0b01 => this.project_index(&left, 1)?,
                            0b10 => this.project_index(&right, 0)?,
                            0b11 => this.project_index(&right, 1)?,
                            _ => unreachable!(),
                        };
                        this.copy_op(&src, &dest)?;
                    }
                }
            }
            // Used to implement the _mm_maskload_ps, _mm_maskload_pd, _mm256_maskload_ps
            // and _mm256_maskload_pd functions.
            // For the element `i`, if the high bit of the `i`-th element of `mask`
            // is one, it is loaded from `ptr.wrapping_add(i)`, otherwise zero is
            // loaded.
            "maskload.ps" | "maskload.pd" | "maskload.ps.256" | "maskload.pd.256" => {
                let [ptr, mask] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                mask_load(this, ptr, mask, dest)?;
            }
            // Used to implement the _mm_maskstore_ps, _mm_maskstore_pd, _mm256_maskstore_ps
            // and _mm256_maskstore_pd functions.
            // For the element `i`, if the high bit of the element `i`-th of `mask`
            // is one, it is stored into `ptr.wapping_add(i)`.
            // Unlike SSE2's _mm_maskmoveu_si128, these are not non-temporal stores.
            "maskstore.ps" | "maskstore.pd" | "maskstore.ps.256" | "maskstore.pd.256" => {
                let [ptr, mask, value] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                mask_store(this, ptr, mask, value)?;
            }
            // Used to implement the _mm256_lddqu_si256 function.
            // Reads a 256-bit vector from an unaligned pointer. This intrinsic
            // is expected to perform better than a regular unaligned read when
            // the data crosses a cache line, but for Miri this is just a regular
            // unaligned read.
            "ldu.dq.256" => {
                let [src_ptr] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let src_ptr = this.read_pointer(src_ptr)?;
                let dest = dest.force_mplace(this)?;

                // Unaligned copy, which is what we want.
                this.mem_copy(src_ptr, dest.ptr(), dest.layout.size, /*nonoverlapping*/ true)?;
            }
            // Used to implement the _mm256_testz_si256, _mm256_testc_si256 and
            // _mm256_testnzc_si256 functions.
            // Tests `op & mask == 0`, `op & mask == mask` or
            // `op & mask != 0 && op & mask != mask`
            "ptestz.256" | "ptestc.256" | "ptestnzc.256" => {
                let [op, mask] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (all_zero, masked_set) = test_bits_masked(this, op, mask)?;
                let res = match unprefixed_name {
                    "ptestz.256" => all_zero,
                    "ptestc.256" => masked_set,
                    "ptestnzc.256" => !all_zero && !masked_set,
                    _ => unreachable!(),
                };

                this.write_scalar(Scalar::from_i32(res.into()), dest)?;
            }
            // Used to implement the _mm256_testz_pd, _mm256_testc_pd, _mm256_testnzc_pd
            // _mm_testz_pd, _mm_testc_pd, _mm_testnzc_pd, _mm256_testz_ps,
            // _mm256_testc_ps, _mm256_testnzc_ps, _mm_testz_ps, _mm_testc_ps and
            // _mm_testnzc_ps functions.
            // Calculates two booleans:
            // `direct`, which is true when the highest bit of each element of `op & mask` is zero.
            // `negated`, which is true when the highest bit of each element of `!op & mask` is zero.
            // Return `direct` (testz), `negated` (testc) or `!direct & !negated` (testnzc)
            "vtestz.pd.256" | "vtestc.pd.256" | "vtestnzc.pd.256" | "vtestz.pd" | "vtestc.pd"
            | "vtestnzc.pd" | "vtestz.ps.256" | "vtestc.ps.256" | "vtestnzc.ps.256"
            | "vtestz.ps" | "vtestc.ps" | "vtestnzc.ps" => {
                let [op, mask] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (direct, negated) = test_high_bits_masked(this, op, mask)?;
                let res = match unprefixed_name {
                    "vtestz.pd.256" | "vtestz.pd" | "vtestz.ps.256" | "vtestz.ps" => direct,
                    "vtestc.pd.256" | "vtestc.pd" | "vtestc.ps.256" | "vtestc.ps" => negated,
                    "vtestnzc.pd.256" | "vtestnzc.pd" | "vtestnzc.ps.256" | "vtestnzc.ps" =>
                        !direct && !negated,
                    _ => unreachable!(),
                };

                this.write_scalar(Scalar::from_i32(res.into()), dest)?;
            }
            // Used to implement the `_mm256_zeroupper` and `_mm256_zeroall` functions.
            // These function clear out the upper 128 bits of all avx registers or
            // zero out all avx registers respectively.
            "vzeroupper" | "vzeroall" => {
                // These functions are purely a performance hint for the CPU.
                // Any registers currently in use will be saved beforehand by the
                // compiler, making these functions no-ops.

                // The only thing that needs to be ensured is the correct calling convention.
                let [] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
