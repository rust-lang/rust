use rustc_abi::{CanonAbi, Size};
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

/// A bitmask constant for scrutinizing the immediate byte provided
/// to the string comparison intrinsics. It distinuishes between
/// 16-bit integers and 8-bit integers. See [`compare_strings`]
/// for more details about the immediate byte.
const USE_WORDS: u8 = 1;

/// A bitmask constant for scrutinizing the immediate byte provided
/// to the string comparison intrinsics. It distinuishes between
/// signed integers and unsigned integers. See [`compare_strings`]
/// for more details about the immediate byte.
const USE_SIGNED: u8 = 2;

/// The main worker for the string comparison intrinsics, where the given
/// strings are analyzed according to the given immediate byte.
///
/// # Arguments
///
/// * `str1` - The first string argument. It is always a length 16 array of bytes
///   or a length 8 array of two-byte words.
/// * `str2` - The second string argument. It is always a length 16 array of bytes
///   or a length 8 array of two-byte words.
/// * `len` is the length values of the supplied strings. It is distinct from the operand length
///   in that it describes how much of `str1` and `str2` will be used for the calculation and may
///   be smaller than the array length of `str1` and `str2`. The string length is counted in bytes
///   if using byte operands and in two-byte words when using two-byte word operands.
///   If the value is `None`, the length of a string is determined by the first
///   null value inside the string.
/// * `imm` is the immediate byte argument supplied to the intrinsic. The byte influences
///   the operation as follows:
///
///   ```text
///   0babccddef
///     || | |||- Use of bytes vs use of two-byte words inside the operation.
///     || | ||
///     || | ||- Use of signed values versus use of unsigned values.
///     || | |
///     || | |- The comparison operation performed. A total of four operations are available.
///     || |    * Equal any: Checks which characters of `str2` are inside `str1`.
///     || |    * String ranges: Check if characters in `str2` are inside the provided character ranges.
///     || |      Adjacent characters in `str1` constitute one range.
///     || |    * String comparison: Mark positions where `str1` and `str2` have the same character.
///     || |    * Substring search: Mark positions where `str1` is a substring in `str2`.
///     || |
///     || |- Result Polarity. The result bits may be subjected to a bitwise complement
///     ||    if these bits are set.
///     ||
///     ||- Output selection. This bit has two meanings depending on the instruction.
///     |   If the instruction is generating a mask, it distinguishes between a bit mask
///     |   and a byte mask. Otherwise it distinguishes between the most significand bit
///     |   and the least significand bit when generating an index.
///     |
///     |- This bit is ignored. It is expected that this bit is set to zero, but it is
///        not a requirement.
///   ```
///
/// # Returns
///
/// A result mask. The bit at index `i` inside the mask is set if 'str2' starting at `i`
/// fulfills the test as defined inside the immediate byte.
/// The mask may be negated if negation flags inside the immediate byte are set.
///
/// For more information, see the Intel Software Developer's Manual, Vol. 2b, Chapter 4.1.
#[expect(clippy::arithmetic_side_effects)]
fn compare_strings<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    str1: &OpTy<'tcx>,
    str2: &OpTy<'tcx>,
    len: Option<(u64, u64)>,
    imm: u8,
) -> InterpResult<'tcx, i32> {
    let default_len = default_len::<u64>(imm);
    let (len1, len2) = if let Some(t) = len {
        t
    } else {
        let len1 = implicit_len(ecx, str1, imm)?.unwrap_or(default_len);
        let len2 = implicit_len(ecx, str2, imm)?.unwrap_or(default_len);
        (len1, len2)
    };

    let mut result = 0;
    match (imm >> 2) & 3 {
        0 => {
            // Equal any: Checks which characters of `str2` are inside `str1`.
            for i in 0..len2 {
                let ch2 = ecx.read_immediate(&ecx.project_index(str2, i)?)?;

                for j in 0..len1 {
                    let ch1 = ecx.read_immediate(&ecx.project_index(str1, j)?)?;

                    let eq = ecx.binary_op(mir::BinOp::Eq, &ch1, &ch2)?;
                    if eq.to_scalar().to_bool()? {
                        result |= 1 << i;
                        break;
                    }
                }
            }
        }
        1 => {
            // String ranges: Check if characters in `str2` are inside the provided character ranges.
            // Adjacent characters in `str1` constitute one range.
            let len1 = len1 - (len1 & 1);
            let get_ch = |ch: Scalar| -> InterpResult<'tcx, i32> {
                let result = match (imm & USE_WORDS != 0, imm & USE_SIGNED != 0) {
                    (true, true) => i32::from(ch.to_i16()?),
                    (true, false) => i32::from(ch.to_u16()?),
                    (false, true) => i32::from(ch.to_i8()?),
                    (false, false) => i32::from(ch.to_u8()?),
                };
                interp_ok(result)
            };

            for i in 0..len2 {
                for j in (0..len1).step_by(2) {
                    let ch2 = get_ch(ecx.read_scalar(&ecx.project_index(str2, i)?)?)?;
                    let ch1_1 = get_ch(ecx.read_scalar(&ecx.project_index(str1, j)?)?)?;
                    let ch1_2 = get_ch(ecx.read_scalar(&ecx.project_index(str1, j + 1)?)?)?;

                    if ch1_1 <= ch2 && ch2 <= ch1_2 {
                        result |= 1 << i;
                    }
                }
            }
        }
        2 => {
            // String comparison: Mark positions where `str1` and `str2` have the same character.
            result = (1 << default_len) - 1;
            result ^= (1 << len1.max(len2)) - 1;

            for i in 0..len1.min(len2) {
                let ch1 = ecx.read_immediate(&ecx.project_index(str1, i)?)?;
                let ch2 = ecx.read_immediate(&ecx.project_index(str2, i)?)?;
                let eq = ecx.binary_op(mir::BinOp::Eq, &ch1, &ch2)?;
                result |= i32::from(eq.to_scalar().to_bool()?) << i;
            }
        }
        3 => {
            // Substring search: Mark positions where `str1` is a substring in `str2`.
            if len1 == 0 {
                result = (1 << default_len) - 1;
            } else if len1 <= len2 {
                for i in 0..len2 {
                    if len1 > len2 - i {
                        break;
                    }

                    result |= 1 << i;

                    for j in 0..len1 {
                        let k = i + j;

                        if k >= default_len {
                            break;
                        } else {
                            let ch1 = ecx.read_immediate(&ecx.project_index(str1, j)?)?;
                            let ch2 = ecx.read_immediate(&ecx.project_index(str2, k)?)?;
                            let ne = ecx.binary_op(mir::BinOp::Ne, &ch1, &ch2)?;

                            if ne.to_scalar().to_bool()? {
                                result &= !(1 << i);
                                break;
                            }
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    // Polarity: Possibly perform a bitwise complement on the result.
    match (imm >> 4) & 3 {
        3 => result ^= (1 << len1) - 1,
        1 => result ^= (1 << default_len) - 1,
        _ => (),
    }

    interp_ok(result)
}

/// Obtain the arguments of the intrinsic based on its name.
/// The result is a tuple with the following values:
/// * The first string argument.
/// * The second string argument.
/// * The string length values, if the intrinsic requires them.
/// * The immediate instruction byte.
///
/// The string arguments will be transmuted into arrays of bytes
/// or two-byte words, depending on the value of the immediate byte.
/// Originally, they are [__m128i](https://doc.rust-lang.org/stable/core/arch/x86_64/struct.__m128i.html) values
/// corresponding to the x86 128-bit integer SIMD type.
fn deconstruct_args<'tcx>(
    unprefixed_name: &str,
    ecx: &mut MiriInterpCx<'tcx>,
    link_name: Symbol,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    args: &[OpTy<'tcx>],
) -> InterpResult<'tcx, (OpTy<'tcx>, OpTy<'tcx>, Option<(u64, u64)>, u8)> {
    let array_layout_fn = |ecx: &mut MiriInterpCx<'tcx>, imm: u8| {
        if imm & USE_WORDS != 0 {
            ecx.layout_of(Ty::new_array(ecx.tcx.tcx, ecx.tcx.types.u16, 8))
        } else {
            ecx.layout_of(Ty::new_array(ecx.tcx.tcx, ecx.tcx.types.u8, 16))
        }
    };

    // The fourth letter of each string comparison intrinsic is either 'e' for "explicit" or 'i' for "implicit".
    // The distinction will correspond to the intrinsics type signature. In this constext, "explicit" and "implicit"
    // refer to the way the string length is determined. The length is either passed explicitly in the "explicit"
    // case or determined by a null terminator in the "implicit" case.
    let is_explicit = match unprefixed_name.as_bytes().get(4) {
        Some(&b'e') => true,
        Some(&b'i') => false,
        _ => unreachable!(),
    };

    if is_explicit {
        let [str1, len1, str2, len2, imm] = ecx.check_shim(abi, CanonAbi::C, link_name, args)?;
        let imm = ecx.read_scalar(imm)?.to_u8()?;

        let default_len = default_len::<u32>(imm);
        let len1 = u64::from(ecx.read_scalar(len1)?.to_u32()?.min(default_len));
        let len2 = u64::from(ecx.read_scalar(len2)?.to_u32()?.min(default_len));

        let array_layout = array_layout_fn(ecx, imm)?;
        let str1 = str1.transmute(array_layout, ecx)?;
        let str2 = str2.transmute(array_layout, ecx)?;

        interp_ok((str1, str2, Some((len1, len2)), imm))
    } else {
        let [str1, str2, imm] = ecx.check_shim(abi, CanonAbi::C, link_name, args)?;
        let imm = ecx.read_scalar(imm)?.to_u8()?;

        let array_layout = array_layout_fn(ecx, imm)?;
        let str1 = str1.transmute(array_layout, ecx)?;
        let str2 = str2.transmute(array_layout, ecx)?;

        interp_ok((str1, str2, None, imm))
    }
}

/// Calculate the c-style string length for a given string `str`.
/// The string is either a length 16 array of bytes a length 8 array of two-byte words.
fn implicit_len<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    str: &OpTy<'tcx>,
    imm: u8,
) -> InterpResult<'tcx, Option<u64>> {
    let mut result = None;
    let zero = ImmTy::from_int(0, str.layout.field(ecx, 0));

    for i in 0..default_len::<u64>(imm) {
        let ch = ecx.read_immediate(&ecx.project_index(str, i)?)?;
        let is_zero = ecx.binary_op(mir::BinOp::Eq, &ch, &zero)?;
        if is_zero.to_scalar().to_bool()? {
            result = Some(i);
            break;
        }
    }
    interp_ok(result)
}

#[inline]
fn default_len<T: From<u8>>(imm: u8) -> T {
    if imm & USE_WORDS != 0 { T::from(8u8) } else { T::from(16u8) }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sse42_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sse4.2")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse42.").unwrap();

        match unprefixed_name {
            // Used to implement the `_mm_cmpestrm` and the `_mm_cmpistrm` functions.
            // These functions compare the input strings and return the resulting mask.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=1044,922
            "pcmpistrm128" | "pcmpestrm128" => {
                let (str1, str2, len, imm) =
                    deconstruct_args(unprefixed_name, this, link_name, abi, args)?;
                let mask = compare_strings(this, &str1, &str2, len, imm)?;

                // The sixth bit inside the immediate byte distiguishes
                // between a bit mask or a byte mask when generating a mask.
                if imm & 0b100_0000 != 0 {
                    let (array_layout, size) = if imm & USE_WORDS != 0 {
                        (this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.u16, 8))?, 2)
                    } else {
                        (this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.u8, 16))?, 1)
                    };
                    let size = Size::from_bytes(size);
                    let dest = dest.transmute(array_layout, this)?;

                    for i in 0..default_len::<u64>(imm) {
                        let result = helpers::bool_to_simd_element(mask & (1 << i) != 0, size);
                        this.write_scalar(result, &this.project_index(&dest, i)?)?;
                    }
                } else {
                    let layout = this.layout_of(this.tcx.types.i128)?;
                    let dest = dest.transmute(layout, this)?;
                    this.write_scalar(Scalar::from_i128(i128::from(mask)), &dest)?;
                }
            }

            // Used to implement the `_mm_cmpestra` and the `_mm_cmpistra` functions.
            // These functions compare the input strings and return `1` if the end of the second
            // input string is not reached and the resulting mask is zero, and `0` otherwise.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=919,1041
            "pcmpistria128" | "pcmpestria128" => {
                let (str1, str2, len, imm) =
                    deconstruct_args(unprefixed_name, this, link_name, abi, args)?;
                let result = if compare_strings(this, &str1, &str2, len, imm)? != 0 {
                    false
                } else if let Some((_, len)) = len {
                    len >= default_len::<u64>(imm)
                } else {
                    implicit_len(this, &str1, imm)?.is_some()
                };

                this.write_scalar(Scalar::from_i32(i32::from(result)), dest)?;
            }

            // Used to implement the `_mm_cmpestri` and the `_mm_cmpistri` functions.
            // These functions compare the input strings and return the bit index
            // for most significant or least significant bit of the resulting mask.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=921,1043
            "pcmpistri128" | "pcmpestri128" => {
                let (str1, str2, len, imm) =
                    deconstruct_args(unprefixed_name, this, link_name, abi, args)?;
                let mask = compare_strings(this, &str1, &str2, len, imm)?;

                let len = default_len::<u32>(imm);
                // The sixth bit inside the immediate byte distiguishes between the least
                // significant bit and the most significant bit when generating an index.
                let result = if imm & 0b100_0000 != 0 {
                    // most significant bit
                    31u32.wrapping_sub(mask.leading_zeros()).min(len)
                } else {
                    // least significant bit
                    mask.trailing_zeros().min(len)
                };
                this.write_scalar(Scalar::from_i32(i32::try_from(result).unwrap()), dest)?;
            }

            // Used to implement the `_mm_cmpestro` and the `_mm_cmpistro` functions.
            // These functions compare the input strings and return the lowest bit of the
            // resulting mask.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=923,1045
            "pcmpistrio128" | "pcmpestrio128" => {
                let (str1, str2, len, imm) =
                    deconstruct_args(unprefixed_name, this, link_name, abi, args)?;
                let mask = compare_strings(this, &str1, &str2, len, imm)?;
                this.write_scalar(Scalar::from_i32(mask & 1), dest)?;
            }

            // Used to implement the `_mm_cmpestrc` and the `_mm_cmpistrc` functions.
            // These functions compare the input strings and return `1` if the resulting
            // mask was non-zero, and `0` otherwise.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=920,1042
            "pcmpistric128" | "pcmpestric128" => {
                let (str1, str2, len, imm) =
                    deconstruct_args(unprefixed_name, this, link_name, abi, args)?;
                let mask = compare_strings(this, &str1, &str2, len, imm)?;
                this.write_scalar(Scalar::from_i32(i32::from(mask != 0)), dest)?;
            }

            // Used to implement the `_mm_cmpistrz` and the `_mm_cmpistrs` functions.
            // These functions return `1` if the string end has been reached and `0` otherwise.
            // Since these functions define the string length implicitly, it is equal to a
            // search for a null terminator (see `deconstruct_args` for more details).
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=924,925
            "pcmpistriz128" | "pcmpistris128" => {
                let [str1, str2, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let imm = this.read_scalar(imm)?.to_u8()?;

                let str = if unprefixed_name == "pcmpistris128" { str1 } else { str2 };
                let array_layout = if imm & USE_WORDS != 0 {
                    this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.u16, 8))?
                } else {
                    this.layout_of(Ty::new_array(this.tcx.tcx, this.tcx.types.u8, 16))?
                };
                let str = str.transmute(array_layout, this)?;
                let result = implicit_len(this, &str, imm)?.is_some();

                this.write_scalar(Scalar::from_i32(i32::from(result)), dest)?;
            }

            // Used to implement the `_mm_cmpestrz` and the `_mm_cmpestrs` functions.
            // These functions return 1 if the explicitly passed string length is smaller
            // than 16 for byte-sized operands or 8 for word-sized operands.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=1046,1047
            "pcmpestriz128" | "pcmpestris128" => {
                let [_, len1, _, len2, imm] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let len = if unprefixed_name == "pcmpestris128" { len1 } else { len2 };
                let len = this.read_scalar(len)?.to_i32()?;
                let imm = this.read_scalar(imm)?.to_u8()?;
                this.write_scalar(
                    Scalar::from_i32(i32::from(len < default_len::<i32>(imm))),
                    dest,
                )?;
            }

            // Used to implement the `_mm_crc32_u{8, 16, 32, 64}` functions.
            // These functions calculate a 32-bit CRC using `0x11EDC6F41`
            // as the polynomial, also known as CRC32C.
            // https://datatracker.ietf.org/doc/html/rfc3720#section-12.1
            "crc32.32.8" | "crc32.32.16" | "crc32.32.32" | "crc32.64.64" => {
                let bit_size = match unprefixed_name {
                    "crc32.32.8" => 8,
                    "crc32.32.16" => 16,
                    "crc32.32.32" => 32,
                    "crc32.64.64" => 64,
                    _ => unreachable!(),
                };

                if bit_size == 64 && this.tcx.sess.target.arch != "x86_64" {
                    return interp_ok(EmulateItemResult::NotSupported);
                }

                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let left = this.read_scalar(left)?;
                let right = this.read_scalar(right)?;

                let crc = if bit_size == 64 {
                    // The 64-bit version will only consider the lower 32 bits,
                    // while the upper 32 bits get discarded.
                    #[expect(clippy::as_conversions)]
                    u128::from((left.to_u64()? as u32).reverse_bits())
                } else {
                    u128::from(left.to_u32()?.reverse_bits())
                };
                let v = match bit_size {
                    8 => u128::from(right.to_u8()?.reverse_bits()),
                    16 => u128::from(right.to_u16()?.reverse_bits()),
                    32 => u128::from(right.to_u32()?.reverse_bits()),
                    64 => u128::from(right.to_u64()?.reverse_bits()),
                    _ => unreachable!(),
                };

                // Perform polynomial division modulo 2.
                // The algorithm for the division is an adapted version of the
                // schoolbook division algorithm used for normal integer or polynomial
                // division. In this context, the quotient is not calculated, since
                // only the remainder is needed.
                //
                // The algorithm works as follows:
                // 1. Pull down digits until division can be performed. In the context of division
                //    modulo 2 it means locating the most significant digit of the dividend and shifting
                //    the divisor such that the position of the divisors most significand digit and the
                //    dividends most significand digit match.
                // 2. Perform a division and determine the remainder. Since it is arithmetic modulo 2,
                //    this operation is a simple bitwise exclusive or.
                // 3. Repeat steps 1. and 2. until the full remainder is calculated. This is the case
                //    once the degree of the remainder polynomial is smaller than the degree of the
                //    divisor polynomial. In other words, the number of leading zeros of the remainder
                //    is larger than the number of leading zeros of the divisor. It is important to
                //    note that standard arithmetic comparison is not applicable here:
                //    0b10011 / 0b11111 = 0b01100 is a valid division, even though the dividend is
                //    smaller than the divisor.
                let mut dividend = (crc << bit_size) ^ (v << 32);
                const POLYNOMIAL: u128 = 0x11EDC6F41;
                while dividend.leading_zeros() <= POLYNOMIAL.leading_zeros() {
                    dividend ^=
                        (POLYNOMIAL << POLYNOMIAL.leading_zeros()) >> dividend.leading_zeros();
                }

                let result = u32::try_from(dividend).unwrap().reverse_bits();
                let result = if bit_size == 64 {
                    Scalar::from_u64(u64::from(result))
                } else {
                    Scalar::from_u32(result)
                };

                this.write_scalar(result, dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
