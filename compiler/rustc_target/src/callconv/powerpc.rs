use rustc_abi::{Reg, RegKind, TyAbiInterface};

use crate::callconv::{ArgAbi, CastTarget, FnAbi, Uniform};
use crate::spec::{Env, HasTargetSpec, Os};

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
}
