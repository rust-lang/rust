use rustc_abi::CanonAbi;
use rustc_apfloat::Float;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use self::math::{ToHost, ToSoft};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // math functions (note that there are also intrinsics for some other functions)
        match link_name.as_str() {
            // math functions (note that there are also intrinsics for some other functions)
            #[rustfmt::skip]
            | "cbrtf"
            | "coshf"
            | "sinhf"
            | "tanf"
            | "tanhf"
            | "acosf"
            | "asinf"
            | "atanf"
            | "acoshf"
            | "asinhf"
            | "log1pf"
            | "expm1f"
            | "tgammaf"
            | "erff"
            | "erfcf"
            => {
                let [f] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f = this.read_scalar(f)?.to_f32()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, these operations do not have
                    // guaranteed precision).
                    let f_host = f.to_host();
                    let res = match link_name.as_str() {
                        "cbrtf" => f_host.cbrt(),
                        "coshf" => f_host.cosh(),
                        "sinhf" => f_host.sinh(),
                        "tanf" => f_host.tan(),
                        "tanhf" => f_host.tanh(),
                        "acosf" => f_host.acos(),
                        "asinf" => f_host.asin(),
                        "atanf" => f_host.atan(),
                        "acoshf" => f_host.acosh(),
                        "asinhf" => f_host.asinh(),
                        "log1pf" => f_host.ln_1p(),
                        "expm1f" => f_host.exp_m1(),
                        "tgammaf" => f_host.gamma(),
                        "erff" => f_host.erf(),
                        "erfcf" => f_host.erfc(),
                        _ => bug!(),
                    };
                    let res = res.to_soft();
                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(this, res, 4);

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(link_name.as_str(), res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "_hypotf"
            | "hypotf"
            | "atan2f"
            | "fdimf"
            => {
                let [f1, f2] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f1 = this.read_scalar(f1)?.to_f32()?;
                let f2 = this.read_scalar(f2)?.to_f32()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f1, f2])
                    .unwrap_or_else(|| {
                        let res = match link_name.as_str() {
                            // underscore case for windows, here and below
                            // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
                            // Using host floats (but it's fine, these operations do not have guaranteed precision).
                            "_hypotf" | "hypotf" => f1.to_host().hypot(f2.to_host()).to_soft(),
                            "atan2f" => f1.to_host().atan2(f2.to_host()).to_soft(),
                            #[allow(deprecated)]
                            "fdimf" => f1.to_host().abs_sub(f2.to_host()).to_soft(),
                            _ => bug!(),
                        };
                        // Apply a relative error of 4ULP to introduce some non-determinism
                        // simulating imprecise implementations and optimizations.
                        let res = math::apply_random_float_error_ulp(this, res, 4);

                        // Clamp the result to the guaranteed range of this function according to the C standard,
                        // if any.
                        math::clamp_float_value(link_name.as_str(), res)
                    });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "cbrt"
            | "cosh"
            | "sinh"
            | "tan"
            | "tanh"
            | "acos"
            | "asin"
            | "atan"
            | "acosh"
            | "asinh"
            | "log1p"
            | "expm1"
            | "tgamma"
            | "erf"
            | "erfc"
            => {
                let [f] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f = this.read_scalar(f)?.to_f64()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, these operations do not have
                    // guaranteed precision).
                    let f_host = f.to_host();
                    let res = match link_name.as_str() {
                        "cbrt" => f_host.cbrt(),
                        "cosh" => f_host.cosh(),
                        "sinh" => f_host.sinh(),
                        "tan" => f_host.tan(),
                        "tanh" => f_host.tanh(),
                        "acos" => f_host.acos(),
                        "asin" => f_host.asin(),
                        "atan" => f_host.atan(),
                        "acosh" => f_host.acosh(),
                        "asinh" => f_host.asinh(),
                        "log1p" => f_host.ln_1p(),
                        "expm1" => f_host.exp_m1(),
                        "tgamma" => f_host.gamma(),
                        "erf" => f_host.erf(),
                        "erfc" => f_host.erfc(),
                        _ => bug!(),
                    };
                    let res = res.to_soft();
                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(this, res, 4);

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(link_name.as_str(), res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "_hypot"
            | "hypot"
            | "atan2"
            | "fdim"
            => {
                let [f1, f2] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;

                let res = math::fixed_float_value(this, link_name.as_str(), &[f1, f2]).unwrap_or_else(|| {
                    let res = match link_name.as_str() {
                        // underscore case for windows, here and below
                        // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
                        // Using host floats (but it's fine, these operations do not have guaranteed precision).
                        "_hypot" | "hypot" => f1.to_host().hypot(f2.to_host()).to_soft(),
                        "atan2" => f1.to_host().atan2(f2.to_host()).to_soft(),
                        #[allow(deprecated)]
                        "fdim" => f1.to_host().abs_sub(f2.to_host()).to_soft(),
                        _ => bug!(),
                    };
                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(this, res, 4);

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(link_name.as_str(), res)
                });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "_ldexp"
            | "ldexp"
            | "scalbn"
            => {
                let [x, exp] = this.check_shim_sig_lenient(abi, CanonAbi::C , link_name, args)?;
                // For radix-2 (binary) systems, `ldexp` and `scalbn` are the same.
                let x = this.read_scalar(x)?.to_f64()?;
                let exp = this.read_scalar(exp)?.to_i32()?;

                let res = x.scalbn(exp);
                let res = this.adjust_nan(res, &[x]);
                this.write_scalar(res, dest)?;
            }
            "lgammaf_r" => {
                let [x, signp] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let x = this.read_scalar(x)?.to_f32()?;
                let signp = this.deref_pointer_as(signp, this.machine.layouts.i32)?;

                // Using host floats (but it's fine, these operations do not have guaranteed precision).
                let (res, sign) = x.to_host().ln_gamma();
                this.write_int(sign, &signp)?;

                let res = res.to_soft();
                // Apply a relative error of 4ULP to introduce some non-determinism
                // simulating imprecise implementations and optimizations.
                let res = math::apply_random_float_error_ulp(this, res, 4);
                // Clamp the result to the guaranteed range of this function according to the C standard,
                // if any.
                let res = math::clamp_float_value(link_name.as_str(), res);
                let res = this.adjust_nan(res, &[x]);
                this.write_scalar(res, dest)?;
            }
            "lgamma_r" => {
                let [x, signp] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let x = this.read_scalar(x)?.to_f64()?;
                let signp = this.deref_pointer_as(signp, this.machine.layouts.i32)?;

                // Using host floats (but it's fine, these operations do not have guaranteed precision).
                let (res, sign) = x.to_host().ln_gamma();
                this.write_int(sign, &signp)?;

                let res = res.to_soft();
                // Apply a relative error of 4ULP to introduce some non-determinism
                // simulating imprecise implementations and optimizations.
                let res = math::apply_random_float_error_ulp(this, res, 4);
                // Clamp the result to the guaranteed range of this function according to the C standard,
                // if any.
                let res = math::clamp_float_value(link_name.as_str(), res);
                let res = this.adjust_nan(res, &[x]);
                this.write_scalar(res, dest)?;
            }

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
