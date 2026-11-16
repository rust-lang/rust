use rustc_abi::{BackendRepr, Float, Primitive, Reg, RegKind, TyAbiInterface};
use rustc_span::bug;

use crate::callconv::{ArgAbi, CastTarget, FnAbi, Uniform};
use crate::spec::{Env, HasTargetSpec, Os, RustcAbi};

const NUM_ARG_GPRS: u64 = 8;

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

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, gprs_in_use: &mut u64)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if ret.layout.is_complex_number(cx) {
        ret.cast_to(complex_cast_target(ret));
    } else if ret.layout.is_aggregate() {
        *gprs_in_use += 1;
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<'a, Ty, C: HasTargetSpec>(cx: &C, arg: &mut ArgAbi<'a, Ty>, gprs_in_use: &mut u64)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.is_ignore() {
        // powerpc-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
        if cx.target_spec().os == Os::Linux
            && matches!(cx.target_spec().env, Env::Gnu | Env::Musl | Env::Uclibc)
            && arg.layout.is_repr_c()
        {
            *gprs_in_use += 1;
            arg.make_indirect_from_ignore();
        }
        return;
    }

    // The powerpc ABI in GCC hardcodes a special rule for values of size 8. It remarks
    //
    // > V.4 wants long longs and doubles to be double word aligned. Just
    // > testing the mode size is a boneheaded way to do this as it means
    // > that other types such as complex int are also double word aligned.
    // > However, we're stuck with this because changing the ABI might break
    // > existing library interfaces.
    //
    // LLVM aligns every value that it splits over multiple GPRs, but a `Complex<T>` is only
    // aligned when it is cast to an `i64` (i.e. has a size of exactly 8).

    if arg.layout.is_complex_number(cx) {
        let cast_target = complex_cast_target(arg);

        // Complex values are only register aligned when the size is exactly 8.
        let is_aligned_pair = arg.layout.size.bytes() == 8;
        if is_aligned_pair && !gprs_in_use.is_multiple_of(2) {
            *gprs_in_use += 1;
        }

        let registers_used = arg.layout.size.bytes().div_ceil(4);
        if *gprs_in_use + registers_used > NUM_ARG_GPRS {
            let pad_i32_count = u8::try_from(NUM_ARG_GPRS.saturating_sub(*gprs_in_use)).unwrap();
            arg.cast_to_and_pad_i32(cast_target, pad_i32_count);
        } else {
            arg.cast_to(cast_target);
        }

        // A value that does not fit uses up the remaining GPRs.
        *gprs_in_use += registers_used;
    } else if arg.layout.pass_indirectly_in_non_rustic_abis(cx) || arg.layout.is_aggregate() {
        *gprs_in_use += 1;
        arg.make_indirect();
    } else {
        if backend_repr_uses_gprs(cx, arg) {
            // Other types (i64, i128, etc.) are register aligned when >= 8.
            let is_aligned_pair = arg.layout.size.bytes() >= 8;
            if is_aligned_pair && !gprs_in_use.is_multiple_of(2) {
                *gprs_in_use += 1;
            }

            *gprs_in_use += arg.layout.size.bytes().div_ceil(4)
        }

        arg.extend_integer_width_to(32);
    }
}

/// Are GPRs used to pass this argument (assuming it is passed directly)?
fn backend_repr_uses_gprs<'a, Ty, C: HasTargetSpec>(cx: &C, arg: &ArgAbi<'a, Ty>) -> bool {
    let is_softfloat = match cx.target_spec().rustc_abi {
        Some(RustcAbi::Softfloat) => bug!("there are currently no powerpc softfloat targets"),
        Some(RustcAbi::PowerPcSpe) => true, // SPE is like softfloat in terms of ABI
        None => false,
        Some(other) => bug!("unexpected powerpc abi {other:?}"),
    };

    match arg.layout.backend_repr {
        BackendRepr::ScalarPair { .. } | BackendRepr::SimdScalableVector { .. } => {
            bug!("unexpected backend_repr for 32-bit powerpc: {:?}", arg.layout.backend_repr)
        }
        BackendRepr::Memory { .. } => {
            bug!("unexpected backend_repr for PassMode::Direct: {:?}", arg.layout.backend_repr)
        }
        BackendRepr::Scalar(scalar) => match scalar.primitive() {
            Primitive::Float(Float::F32 | Float::F64 | Float::F128 | Float::PpcF128) => {
                is_softfloat
            }

            _ => true,
        },
        BackendRepr::SimdVector { .. } => {
            // When vector types are legal, they are passed in vector registers.
            false
        }
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C: HasTargetSpec>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    // Keeps track of how many GPRs are in use so far to pass arguments. This is used to exhaust
    // the remaining argument registers when a complex value, that doesn't fit fully in the
    // remaining registers, is passed via the stack. Subsequent arguments must then also be passed
    // via the stack.
    let mut gprs_in_use: u64 = 0;

    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, &mut gprs_in_use);
    }

    for arg in fn_abi.args.iter_mut() {
        classify_arg(cx, arg, &mut gprs_in_use);
    }
}
