// see https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/BPF/BPFCallingConv.td
use rustc_abi::{Reg, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Uniform};

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if !ret.layout.is_sized() {
        return;
    }

    let size = ret.layout.size;
    if size.bits() > 128 {
        ret.make_indirect();
        return;
    }

    if ret.layout.is_aggregate() {
        // Return small aggregates in up to two 64-bit registers.
        ret.cast_to(Uniform::new(Reg::i64(), size));
        return;
    }

    ret.extend_integer_width_to(32);
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if arg.layout.is_aggregate() || arg.layout.size.bits() > 64 {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}

pub(crate) fn compute_rust_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>) {
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }
}
