use rustc_apfloat::ieee::{DoubleS, HalfS, IeeeFloat, Semantics, SingleS};
use rustc_apfloat::{self, Float, FloatConvert, Round};
use rustc_middle::mir;
use rustc_middle::ty::{self, FloatTy};

use self::math::{HostFloatOperation, HostUnaryFloatOp, IeeeExt, host_unary_float_op};
use super::check_intrinsic_arg_count;
use crate::*;

fn sqrt<'tcx, F: Float + FloatConvert<F> + Into<Scalar>>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &PlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    let [f] = check_intrinsic_arg_count(args)?;
    math::sqrt_op::<F>(this, f, dest)
}

/// Determine which float operation on which type this is.
fn is_host_unary_float_op(
    intrinsic_name: &str,
    generic_args: ty::GenericArgsRef<'_>,
) -> Option<(FloatTy, HostUnaryFloatOp)> {
    let host_float_op = match intrinsic_name {
        "sin" => HostUnaryFloatOp::Sin,
        "cos" => HostUnaryFloatOp::Cos,
        "exp" => HostUnaryFloatOp::Exp,
        "exp2" => HostUnaryFloatOp::Exp2,
        "log" => HostUnaryFloatOp::Log,
        "log10" => HostUnaryFloatOp::Log10,
        "log2" => HostUnaryFloatOp::Log2,
        _ => return None,
    };

    let ty::Float(float_ty) = *generic_args.type_at(0).kind() else {
        bug!("`{intrinsic_name}` intrinsic called on non-float type");
    };
    Some((float_ty, host_float_op))
}

fn pow_intrinsic<'tcx, S: Semantics>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &PlaceTy<'tcx>,
) -> InterpResult<'tcx, ()>
where
    IeeeFloat<S>: HostFloatOperation + IeeeExt + Float + Into<Scalar>,
{
    let [f1, f2] = check_intrinsic_arg_count(args)?;
    let f1: IeeeFloat<S> = this.read_scalar(f1)?.to_float()?;
    let f2: IeeeFloat<S> = this.read_scalar(f2)?.to_float()?;

    let res = math::fixed_float_value(this, "pow", &[f1, f2]).unwrap_or_else(|| {
        // Using host floats (but it's fine, this operation does not have guaranteed precision).
        let res = f1.host_powf(f2);

        // Apply a relative error of 4ULP to introduce some non-determinism
        // simulating imprecise implementations and optimizations.
        math::apply_random_float_error_ulp(this, res, 4)
    });
    let res = this.adjust_nan(res, &[f1, f2]);
    this.write_scalar(res, dest)?;
    interp_ok(())
}
fn powi_intrinsic<'tcx, S: Semantics>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &PlaceTy<'tcx>,
) -> InterpResult<'tcx, ()>
where
    IeeeFloat<S>: HostFloatOperation + IeeeExt + Float + Into<Scalar>,
{
    let [f, i] = check_intrinsic_arg_count(args)?;
    let f: IeeeFloat<S> = this.read_scalar(f)?.to_float()?;
    let i = this.read_scalar(i)?.to_i32()?;

    let res = math::fixed_powi_value(this, f, i).unwrap_or_else(|| {
        // Using host floats (but it's fine, this operation does not have guaranteed precision).
        let res = f.host_powi(i);

        // Apply a relative error of 4ULP to introduce some non-determinism
        // simulating imprecise implementations and optimizations.
        math::apply_random_float_error_ulp(this, res, 4)
    });
    let res = this.adjust_nan(res, &[f]);
    this.write_scalar(res, dest)?;
    interp_ok(())
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_math_intrinsic(
        &mut self,
        intrinsic_name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        match intrinsic_name {
            // Operations we can do with soft-floats.
            "sqrt" => {
                let ty::Float(float_ty) = *generic_args.type_at(0).kind() else {
                    bug!("`sqrt` intrinsic called on non-float type");
                };
                match float_ty {
                    FloatTy::F16 => sqrt::<rustc_apfloat::ieee::Half>(this, args, dest)?,
                    FloatTy::F32 => sqrt::<rustc_apfloat::ieee::Single>(this, args, dest)?,
                    FloatTy::F64 => sqrt::<rustc_apfloat::ieee::Double>(this, args, dest)?,
                    FloatTy::F128 => sqrt::<rustc_apfloat::ieee::Quad>(this, args, dest)?,
                }
            }

            #[rustfmt::skip]
            | "fadd_fast"
            | "fsub_fast"
            | "fmul_fast"
            | "fdiv_fast"
            | "frem_fast"
            => {
                let [a, b] = check_intrinsic_arg_count(args)?;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;
                let op = match intrinsic_name {
                    "fadd_fast" => mir::BinOp::Add,
                    "fsub_fast" => mir::BinOp::Sub,
                    "fmul_fast" => mir::BinOp::Mul,
                    "fdiv_fast" => mir::BinOp::Div,
                    "frem_fast" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                let float_finite = |x: &ImmTy<'tcx>| -> InterpResult<'tcx, bool> {
                    let ty::Float(fty) = x.layout.ty.kind() else {
                        bug!("float_finite: non-float input type {}", x.layout.ty)
                    };
                    interp_ok(match fty {
                        FloatTy::F16 => x.to_scalar().to_f16()?.is_finite(),
                        FloatTy::F32 => x.to_scalar().to_f32()?.is_finite(),
                        FloatTy::F64 => x.to_scalar().to_f64()?.is_finite(),
                        FloatTy::F128 => x.to_scalar().to_f128()?.is_finite(),
                    })
                };
                match (float_finite(&a)?, float_finite(&b)?) {
                    (false, false) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as both parameters",
                    ),
                    (false, _) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as first parameter",
                    ),
                    (_, false) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as second parameter",
                    ),
                    _ => {}
                }
                let res = this.binary_op(op, &a, &b)?;
                // This cannot be a NaN so we also don't have to apply any non-determinism.
                // (Also, `binary_op` already called `generate_nan` if needed.)
                if !float_finite(&res)? {
                    throw_ub_format!("`{intrinsic_name}` intrinsic produced non-finite value as result");
                }
                // Apply a relative error of 4ULP to simulate non-deterministic precision loss
                // due to optimizations.
                let res = math::apply_random_float_error_to_imm(this, res, 4)?;
                this.write_immediate(*res, dest)?;
            }

            "float_to_int_unchecked" => {
                let [val] = check_intrinsic_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = this
                    .float_to_int_checked(&val, dest.layout, Round::TowardZero)?
                    .ok_or_else(|| {
                        err_ub_format!(
                            "`float_to_int_unchecked` intrinsic called on {val} which cannot be represented in target type `{:?}`",
                            dest.layout.ty
                        )
                    })?;

                this.write_immediate(*res, dest)?;
            }

            // Operations that need host floats.
            _ if let Some((float_ty, op)) =
                is_host_unary_float_op(intrinsic_name, generic_args) =>
            {
                let [f] = check_intrinsic_arg_count(args)?;
                match float_ty {
                    FloatTy::F16 => host_unary_float_op::<HalfS>(this, f, op, dest)?,
                    FloatTy::F32 => host_unary_float_op::<SingleS>(this, f, op, dest)?,
                    FloatTy::F64 => host_unary_float_op::<DoubleS>(this, f, op, dest)?,
                    FloatTy::F128 => todo!("f128"), // FIXME(f128)
                };
            }

            "powf" => {
                let ty::Float(float_ty) = *generic_args.type_at(0).kind() else {
                    bug!("`powf` intrinsic called on non-float type");
                };
                match float_ty {
                    FloatTy::F16 => pow_intrinsic::<HalfS>(this, args, dest)?,
                    FloatTy::F32 => pow_intrinsic::<SingleS>(this, args, dest)?,
                    FloatTy::F64 => pow_intrinsic::<DoubleS>(this, args, dest)?,
                    FloatTy::F128 => todo!("f128"), // FIXME(f128)
                }
            }

            "powif16" => powi_intrinsic::<HalfS>(this, args, dest)?,
            "powif32" => powi_intrinsic::<SingleS>(this, args, dest)?,
            "powif64" => powi_intrinsic::<DoubleS>(this, args, dest)?,
            "powif128" => todo!("f128"), // FIXME(f128)

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

/// Compute a CRC32 checksum using the given polynomial.
///
/// `bit_size` is the number of relevant data bits (8, 16, 32, or 64).
/// Only the low `bit_size` bits of `data` are used; higher bits must be zero.
/// `polynomial` includes the leading 1 bit (e.g. `0x11EDC6F41` for CRC32C).
///
/// Following hardware CRC conventions, `crc` and `data` bits are assumed to be reversed,
/// and output bits will be equally reversed.
pub(crate) fn compute_crc32(crc: u32, data: u64, bit_size: u32, polynomial: u128) -> u32 {
    assert!(
        bit_size == 64 || data < 1u64.strict_shl(bit_size),
        "crc32: `data` is larger than {bit_size} bits"
    );
    // Bit-reverse inputs to match hardware CRC conventions.
    let crc = u128::from(crc.reverse_bits());
    // Reverse all 64 bits of `data`, then shift right by `64 - bit_size`. This
    // discards the (now-reversed) higher bits, leaving only the reversed low
    // `bit_size` bits in the lowest positions (with zeros above).
    let v = u128::from(data.reverse_bits() >> (64u32.strict_sub(bit_size)));

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
    while dividend.leading_zeros() <= polynomial.leading_zeros() {
        dividend ^= (polynomial << polynomial.leading_zeros()) >> dividend.leading_zeros();
    }

    u32::try_from(dividend).unwrap().reverse_bits()
}

/// AES primitives
pub(crate) mod aes {
    /// AES S-box
    ///
    /// Source: [NIST Advanced Encryption Standar][1], Figure 7 (page 16)
    ///
    /// [1]: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=901427
    const SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
        0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
        0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
        0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
        0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
        0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
        0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
        0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
        0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
        0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
        0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
        0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
        0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
        0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
        0x16,
    ];

    /// Applies S-box substitution to each byte of a 32-bit word
    pub(crate) fn sub_word(word: u32) -> u32 {
        let bytes = word.to_ne_bytes().map(|b| SBOX[usize::from(b)]);
        u32::from_ne_bytes(bytes)
    }
}

// sha256 primitives shared by the x86 and aarch64 intrinsics. Math helpers adapted from RustCrypto soft impl:
// https://github.com/RustCrypto/hashes/blob/3d2bc57db40fd6aeb25d6c6da98d67e2784c2985/sha2/src/sha256/soft/compact.rs
pub(crate) mod sha256 {
    pub(crate) fn sigma0(x: u32) -> u32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }

    pub(crate) fn sigma1(x: u32) -> u32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }

    /// One round of the compression; `wk` is the round's `w[i] + k[i]`.
    pub(crate) fn round(state: [u32; 8], wk: u32) -> [u32; 8] {
        let [a, b, c, d, e, f, g, h] = state;

        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = s1.wrapping_add(ch).wrapping_add(wk).wrapping_add(h);

        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);

        [
            t1.wrapping_add(t2), // a
            a,                   // b
            b,                   // c
            c,                   // d
            d.wrapping_add(t1),  // e
            e,                   // f
            f,                   // g
            g,                   // h
        ]
    }
}
