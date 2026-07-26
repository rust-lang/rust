use rustc_abi::{Reg, RegKind, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Uniform};
use crate::spec::{Env, HasTargetSpec, Os};

fn classify_complex<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    // NOTE: we follow GCC, not Clang here, see https://github.com/llvm/llvm-project/pull/208917.
    let size = arg.layout.size;
    if size.bytes() <= 4 {
        arg.cast_to(Reg { kind: RegKind::Integer, size });
    } else {
        arg.cast_to(Uniform::new(Reg::i32(), size));
    }
}

fn classify_ret<'a, Ty, C>(_cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if ret.layout.is_complex() {
        classify_complex(ret);
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
            && arg.layout.is_zst()
        {
            arg.make_indirect_from_ignore();
        }
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
    } else if arg.layout.is_complex() {
        classify_complex(arg);
    } else if arg.layout.is_aggregate() {
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
