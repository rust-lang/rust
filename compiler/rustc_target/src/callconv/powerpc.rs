use rustc_abi::{BackendRepr, Float, Integer, Primitive, Reg, RegKind, TyAbiInterface};
use rustc_span::bug;

use super::PassMode;
use crate::callconv::{ArgAbi, CastTarget, FnAbi, Uniform};
use crate::spec::{Env, HasTargetSpec, Os, RustcAbi};

/// How to cast `Complex<T>` so that we match the GCC ABI.
fn complex_cast_target<Ty>(arg: &ArgAbi<'_, Ty>) -> CastTarget {
    let size = arg.layout.size;

    if size.bytes() <= 4 {
        // Coerce to an integer for `Complex<i8>` and `Complex<i16>`.
        CastTarget::from(Reg { kind: RegKind::Integer, size })
    } else if size.bytes() == 8 {
        // Coerce to a single `i64` for `Complex<f32>` and `Complex<i32>`, which has the correct
        // register alignment of 8 bytes.
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

fn classify_arg<'a, Ty, C: HasTargetSpec>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.is_ignore() {
        // powerpc-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
        if cx.target_spec().os == Os::Linux
            && matches!(cx.target_spec().env, Env::Gnu | Env::Musl | Env::Uclibc)
            && arg.layout.is_repr_c()
        {
            arg.make_indirect_from_ignore();
        }
        return;
    }
    if arg.layout.is_complex_number(cx) {
        arg.cast_to(complex_cast_target(arg));
    } else if arg.layout.pass_indirectly_in_non_rustic_abis(cx) || arg.layout.is_aggregate() {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C: HasTargetSpec>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        classify_arg(cx, arg);
    }

    // A second pass that adds padding to complex arguments.
    pad_complex_arguments(cx, fn_abi);
}

struct GprUsage {
    /// GPRs used by this argument.
    registers_used: u64,
    /// Whether the argument must be in an even-odd register pair.
    is_aligned_pair: bool,
}

/// Determine the GPR usage of this argument.
fn determine_gpr_usage<'a, Ty, C: HasTargetSpec>(cx: &C, arg: &ArgAbi<'a, Ty>) -> Option<GprUsage>
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    // The "spe" targets are softfloat targets.
    let is_softfloat = match cx.target_spec().rustc_abi {
        Some(RustcAbi::PowerPcSpe) => true,
        None => false,
        Some(other) => bug!("unexpected powerpc ABI {other:?}"),
    };

    // The powerpc ABI in GCC hardcodes a special rule for values of size 8. It remarks
    //
    // > V.4 wants long longs and doubles to be double word aligned. Just
    // > testing the mode size is a boneheaded way to do this as it means
    // > that other types such as complex int are also double word aligned.
    // > However, we're stuck with this because changing the ABI might break
    // > existing library interfaces.
    //
    // LLVM aligns every value that it splits over multiple GPRs, but a `Complex<T>` is only
    // aligned when it is cast to an `i64` (i.e. has a size of 8).

    let default = GprUsage {
        registers_used: arg.layout.size.bytes().div_ceil(4),
        is_aligned_pair: arg.layout.size.bytes() >= 8,
    };

    // Deduce the number of GPRs from the PassMode that we've calculated.
    match arg.mode {
        PassMode::Ignore => None,
        PassMode::Indirect { .. } => {
            // A thin pointer uses a single GPR.
            Some(GprUsage { registers_used: 1, is_aligned_pair: false })
        }
        PassMode::IndirectUnsized { .. } => {
            // A fat pointer uses two GPRs.
            Some(GprUsage { registers_used: 2, is_aligned_pair: false })
        }
        PassMode::Pair(..) => bug!("powerpc32 does not use a scalar pair pass mode"),
        PassMode::Cast { pad_i32_count, cast: _ } => {
            // Nothing should have added padding yet.
            assert_eq!(pad_i32_count, 0);

            // Only Complex<T> uses PassMode::Cast.
            assert!(arg.layout.is_complex_number(cx));

            // A `Complex<T>` is only aligned when it is cast to an `i64` (i.e. has a size of 8).
            Some(GprUsage {
                registers_used: default.registers_used,
                is_aligned_pair: arg.layout.size.bytes() == 8,
            })
        }
        PassMode::Direct(_) => match arg.layout.backend_repr {
            BackendRepr::ScalarPair { .. } | BackendRepr::SimdScalableVector { .. } => {
                bug!("unexpected backend_repr for 32-bit powerpc: {:?}", arg.layout.backend_repr)
            }
            BackendRepr::Memory { .. } => {
                bug!("unexpected backend_repr for PassMode::Direct: {:?}", arg.layout.backend_repr)
            }
            BackendRepr::Scalar(scalar) => match scalar.primitive() {
                Primitive::Float(Float::F32 | Float::F64 | Float::PpcF128) if !is_softfloat => None,
                _ => Some(default),
            },
            BackendRepr::SimdVector { element, count } => match element.primitive() {
                // A single-element vector is scalarized.
                Primitive::Float(Float::F32 | Float::F64)
                    if count.as_u32() == 1 && !is_softfloat =>
                {
                    None
                }
                _ if count.as_u32() == 1 => Some(default),

                // Vectors of i64 element type are only supported with VSX.
                Primitive::Int(Integer::I64, _) => Some(default),

                // Any other vector is passed in vector registers, not in GPRs.
                _ => None,
            },
        },
    }
}

/// Add padding to `Complex<T>` arguments so that they are never split between registers and the
/// stack. When a complex argument is pushed to the stack, remaining argument registers are
/// consumed, so any subsequent arguments are also passed on the stack.
fn pad_complex_arguments<'a, Ty, C: HasTargetSpec>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    const NUM_ARG_GPRS: u64 = 8;

    // The sret pointer is passed in a GPR.
    let mut gprs_in_use: u64 = if fn_abi.ret.is_indirect() { 1 } else { 0 };

    for arg in fn_abi.args.iter_mut() {
        let Some(gpr_usage) = determine_gpr_usage(cx, arg) else {
            // This argument uses no GPRs.
            continue;
        };

        if gpr_usage.is_aligned_pair && !gprs_in_use.is_multiple_of(2) {
            gprs_in_use += 1;
        }

        if arg.layout.is_complex_number(cx)
            && gprs_in_use + gpr_usage.registers_used > NUM_ARG_GPRS
            && let PassMode::Cast { pad_i32_count, .. } = &mut arg.mode
        {
            *pad_i32_count = u8::try_from(NUM_ARG_GPRS - gprs_in_use).unwrap();
        }

        // A value that does not fit uses up the remaining GPRs.
        gprs_in_use = Ord::min(gprs_in_use + gpr_usage.registers_used, NUM_ARG_GPRS);
    }
}
