use rustc_middle::mir;
use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateForeignItemResult;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub(super) trait EvalContextExt<'mir, 'tcx: 'mir>:
    crate::MiriInterpCxExt<'mir, 'tcx>
{
    fn emulate_x86_sse41_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateForeignItemResult> {
        let this = self.eval_context_mut();
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
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

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
                        this.copy_op(
                            &this.project_index(&left, i)?,
                            &dest,
                            /*allow_transmute*/ false,
                        )?;
                    }
                }
            }
            // Used to implement the _mm_packus_epi32 function.
            // Concatenates two 32-bit signed integer vectors and converts
            // the result to a 16-bit unsigned integer vector with saturation.
            "packusdw" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(dest_len, left_len.checked_mul(2).unwrap());

                for i in 0..left_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_i32()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_i32()?;
                    let left_dest = this.project_index(&dest, i)?;
                    let right_dest = this.project_index(&dest, i.checked_add(left_len).unwrap())?;

                    let left_res =
                        u16::try_from(left).unwrap_or(if left < 0 { 0 } else { u16::MAX });
                    let right_res =
                        u16::try_from(right).unwrap_or(if right < 0 { 0 } else { u16::MAX });

                    this.write_scalar(Scalar::from_u16(left_res), &left_dest)?;
                    this.write_scalar(Scalar::from_u16(right_res), &right_dest)?;
                }
            }
            // Used to implement the _mm_dp_ps and _mm_dp_pd functions.
            // Conditionally multiplies the packed floating-point elements in
            // `left` and `right` using the high 4 bits in `imm`, sums the four
            // products, and conditionally stores the sum in `dest` using the low
            // 4 bits of `imm`.
            "dpps" | "dppd" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert!(dest_len <= 4);

                let imm = this.read_scalar(imm)?.to_u8()?;

                let element_layout = left.layout.field(this, 0);

                // Calculate dot product
                // Elements are floating point numbers, but we can use `from_int`
                // because the representation of 0.0 is all zero bits.
                let mut sum = ImmTy::from_int(0u8, element_layout);
                for i in 0..left_len {
                    if imm & (1 << i.checked_add(4).unwrap()) != 0 {
                        let left = this.read_immediate(&this.project_index(&left, i)?)?;
                        let right = this.read_immediate(&this.project_index(&right, i)?)?;

                        let mul = this.wrapping_binary_op(mir::BinOp::Mul, &left, &right)?;
                        sum = this.wrapping_binary_op(mir::BinOp::Add, &sum, &mul)?;
                    }
                }

                // Write to destination (conditioned to imm)
                for i in 0..dest_len {
                    let dest = this.project_index(&dest, i)?;

                    if imm & (1 << i) != 0 {
                        this.write_immediate(*sum, &dest)?;
                    } else {
                        this.write_scalar(Scalar::from_int(0u8, element_layout.size), &dest)?;
                    }
                }
            }
            // Used to implement the _mm_floor_ss, _mm_ceil_ss and _mm_round_ss
            // functions. Rounds the first element of `right` according to `rounding`
            // and copies the remaining elements from `left`.
            "round.ss" => {
                let [left, right, rounding] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                round_first::<rustc_apfloat::ieee::Single>(this, left, right, rounding, dest)?;
            }
            // Used to implement the _mm_floor_ps, _mm_ceil_ps and _mm_round_ps
            // functions. Rounds the elements of `op` according to `rounding`.
            "round.ps" => {
                let [op, rounding] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                round_all::<rustc_apfloat::ieee::Single>(this, op, rounding, dest)?;
            }
            // Used to implement the _mm_floor_sd, _mm_ceil_sd and _mm_round_sd
            // functions. Rounds the first element of `right` according to `rounding`
            // and copies the remaining elements from `left`.
            "round.sd" => {
                let [left, right, rounding] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                round_first::<rustc_apfloat::ieee::Double>(this, left, right, rounding, dest)?;
            }
            // Used to implement the _mm_floor_pd, _mm_ceil_pd and _mm_round_pd
            // functions. Rounds the elements of `op` according to `rounding`.
            "round.pd" => {
                let [op, rounding] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                round_all::<rustc_apfloat::ieee::Double>(this, op, rounding, dest)?;
            }
            // Used to implement the _mm_minpos_epu16 function.
            // Find the minimum unsinged 16-bit integer in `op` and
            // returns its value and position.
            "phminposuw" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

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
                // Fill remaining with zeros
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
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(left_len, dest_len.checked_mul(2).unwrap());

                let imm = this.read_scalar(imm)?.to_u8()?;
                // Bit 2 of `imm` specifies the offset for indices of `left`.
                // The offset is 0 when the bit is 0 or 4 when the bit is 1.
                let left_offset = u64::from((imm >> 2) & 1).checked_mul(4).unwrap();
                // Bits 0..=1 of `imm` specify the offset for indices of
                // `right` in blocks of 4 elements.
                let right_offset = u64::from(imm & 0b11).checked_mul(4).unwrap();

                for i in 0..dest_len {
                    let left_offset = left_offset.checked_add(i).unwrap();
                    let mut res: u16 = 0;
                    for j in 0..4 {
                        let left = this
                            .read_scalar(
                                &this.project_index(&left, left_offset.checked_add(j).unwrap())?,
                            )?
                            .to_u8()?;
                        let right = this
                            .read_scalar(
                                &this
                                    .project_index(&right, right_offset.checked_add(j).unwrap())?,
                            )?
                            .to_u8()?;
                        res = res.checked_add(left.abs_diff(right).into()).unwrap();
                    }
                    this.write_scalar(Scalar::from_u16(res), &this.project_index(&dest, i)?)?;
                }
            }
            // Used to implement the _mm_testz_si128, _mm_testc_si128
            // and _mm_testnzc_si128 functions.
            // Tests `op & mask == 0`, `op & mask == mask` or
            // `op & mask != 0 && op & mask != mask`
            "ptestz" | "ptestc" | "ptestnzc" => {
                let [op, mask] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (op, op_len) = this.operand_to_simd(op)?;
                let (mask, mask_len) = this.operand_to_simd(mask)?;

                assert_eq!(op_len, mask_len);

                let f = match unprefixed_name {
                    "ptestz" => |op, mask| op & mask == 0,
                    "ptestc" => |op, mask| op & mask == mask,
                    "ptestnzc" => |op, mask| op & mask != 0 && op & mask != mask,
                    _ => unreachable!(),
                };

                let mut all_zero = true;
                for i in 0..op_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?.to_u64()?;
                    let mask = this.read_scalar(&this.project_index(&mask, i)?)?.to_u64()?;
                    all_zero &= f(op, mask);
                }

                this.write_scalar(Scalar::from_i32(all_zero.into()), dest)?;
            }
            _ => return Ok(EmulateForeignItemResult::NotSupported),
        }
        Ok(EmulateForeignItemResult::NeedsJumping)
    }
}

// Rounds the first element of `right` according to `rounding`
// and copies the remaining elements from `left`.
fn round_first<'tcx, F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    rounding: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    let rounding = rounding_from_imm(this.read_scalar(rounding)?.to_i32()?)?;

    let op0: F = this.read_scalar(&this.project_index(&right, 0)?)?.to_float()?;
    let res = op0.round_to_integral(rounding).value;
    this.write_scalar(
        Scalar::from_uint(res.to_bits(), Size::from_bits(F::BITS)),
        &this.project_index(&dest, 0)?,
    )?;

    for i in 1..dest_len {
        this.copy_op(
            &this.project_index(&left, i)?,
            &this.project_index(&dest, i)?,
            /*allow_transmute*/ false,
        )?;
    }

    Ok(())
}

// Rounds all elements of `op` according to `rounding`.
fn round_all<'tcx, F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    op: &OpTy<'tcx, Provenance>,
    rounding: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, op_len);

    let rounding = rounding_from_imm(this.read_scalar(rounding)?.to_i32()?)?;

    for i in 0..dest_len {
        let op: F = this.read_scalar(&this.project_index(&op, i)?)?.to_float()?;
        let res = op.round_to_integral(rounding).value;
        this.write_scalar(
            Scalar::from_uint(res.to_bits(), Size::from_bits(F::BITS)),
            &this.project_index(&dest, i)?,
        )?;
    }

    Ok(())
}

/// Gets equivalent `rustc_apfloat::Round` from rounding mode immediate of
/// `round.{ss,sd,ps,pd}` intrinsics.
fn rounding_from_imm<'tcx>(rounding: i32) -> InterpResult<'tcx, rustc_apfloat::Round> {
    // The fourth bit of `rounding` only affects the SSE status
    // register, which cannot be accessed from Miri (or from Rust,
    // for that matter), so we can ignore it.
    match rounding & !0b1000 {
        // When the third bit is 0, the rounding mode is determined by the
        // first two bits.
        0b000 => Ok(rustc_apfloat::Round::NearestTiesToEven),
        0b001 => Ok(rustc_apfloat::Round::TowardNegative),
        0b010 => Ok(rustc_apfloat::Round::TowardPositive),
        0b011 => Ok(rustc_apfloat::Round::TowardZero),
        // When the third bit is 1, the rounding mode is determined by the
        // SSE status register. Since we do not support modifying it from
        // Miri (or Rust), we assume it to be at its default mode (round-to-nearest).
        0b100..=0b111 => Ok(rustc_apfloat::Round::NearestTiesToEven),
        rounding => throw_unsup_format!("unsupported rounding mode 0x{rounding:02x}"),
    }
}
