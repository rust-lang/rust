use rustc_abi::{BackendRepr, Primitive, Reg, RegKind, TyAbiInterface};

use crate::callconv::{ArgAbi, CastTarget, FnAbi, Uniform};
use crate::spec::{Env, HasTargetSpec, Os};

const NUM_ARG_GPRS: u32 = 8; // r3..=r10

/// How to cast `Complex<T>` so that we match the GCC ABI.
fn complex_cast_target<Ty>(arg: &ArgAbi<'_, Ty>) -> CastTarget {
    let size = arg.layout.size;

    if size.bytes() <= 4 {
        // Coerce to an integer for `Complex<i8>` and `Complex<i16>`.
        CastTarget::from(Reg { kind: RegKind::Integer, size })
    } else if size.bytes() == 8 {
        // Coerce to a single `i64` for `Complex<f32>` and `Complex<i32>`, which has the correct
        // register alignment of 8 bytes.
        //
        // NOTE: clang uses a vector (e.g. <2 x f32>) here, but if we try that we run into
        // ABI issues because vectors require the altivec target feature.
        CastTarget::from(Reg::i64())
    } else {
        // Coerce to an array `[N x i32]` for everything wider. An array of i32 gives the correct
        // 4-byte register alignment.
        CastTarget::from(Uniform::new(Reg::i32(), size))
    }
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if ret.layout.is_complex_number(cx) {
        ret.cast_to(complex_cast_target(ret));
    } else if ret.layout.is_aggregate() {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<'a, Ty, C: HasTargetSpec>(cx: &C, arg: &mut ArgAbi<'a, Ty>, arg_gprs_left: &mut u32)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.is_ignore() {
        // powerpc-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
        if cx.target_spec().os == Os::Linux
            && matches!(cx.target_spec().env, Env::Gnu | Env::Musl | Env::Uclibc)
            && arg.layout.is_zst()
        {
            arg.make_indirect_from_ignore();
        }
        return;
    }

    let default = |arg: &mut ArgAbi<'a, Ty>| {
        if arg.layout.pass_indirectly_in_non_rustic_abis(cx) || arg.layout.is_aggregate() {
            arg.make_indirect();
        } else {
            arg.extend_integer_width_to(32);
        }
    };

    let is_complex = arg.layout.is_complex_number(cx);
    let is_float = match arg.layout.backend_repr {
        BackendRepr::Scalar(scalar) => matches!(scalar.primitive(), Primitive::Float(_)),
        _ => false,
    };

    // Arguments that are not relevant for the GPR budget: floats go in the FPRs, and once the GPRs
    // are exhausted everything lands on the stack anyway. Complex<T> always needs custom handling.
    if (*arg_gprs_left == 0 || is_float) && !is_complex {
        return default(arg);
    }

    let size = arg.layout.size;
    let regs_needed = size.bytes().div_ceil(4) as u32; // 32-bit registers

    if arg.layout.is_aggregate() && !is_complex {
        // Non-complex aggregates are passed indirectly, and consume one GPR.
        *arg_gprs_left -= 1;
    } else {
        let mut padding = 0;

        // The powerpc ABI in GCC hardcodes a special rule for values of size 8. It remarks
        //
        // > V.4 wants long longs and doubles to be double word aligned. Just
        // > testing the mode size is a boneheaded way to do this as it means
        // > that other types such as complex int are also double word aligned.
        // > However, we're stuck with this because changing the ABI might break
        // > existing library interfaces.
        //
        // An eight-byte value must start in an even-numbered GPR. The `i64` it is coerced to
        // already makes LLVM skip an odd register, so only account for it in the budget.
        if size.bytes() == 8 && !arg_gprs_left.is_multiple_of(2) {
            *arg_gprs_left -= 1;
        }

        if regs_needed <= *arg_gprs_left {
            // Everything fits, great!
            *arg_gprs_left -= regs_needed;
        } else if is_complex {
            // Never split a Complex<T> across the GPRs and the stack.
            //
            // The full complex value is passed via the stack, and the remaining GPRs are consumed,
            // so all subsequent arguments will also be passed via the stack. Use the padding value
            // to fill up the remaining GPRs.
            padding += *arg_gprs_left;
            *arg_gprs_left = 0;
        }

        if is_complex {
            arg.cast_to_and_pad_i32(complex_cast_target(arg), padding as u8);
            return;
        }
    }

    default(arg)
}

pub(crate) fn compute_abi_info<'a, Ty, C: HasTargetSpec>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    let mut arg_gprs_left = NUM_ARG_GPRS;
    for arg in fn_abi.args.iter_mut() {
        classify_arg(cx, arg, &mut arg_gprs_left);
    }
}
