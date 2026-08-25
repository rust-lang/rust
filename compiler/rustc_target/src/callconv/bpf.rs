// see https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/BPF/BPFCallingConv.td
use rustc_abi::{Reg, RegKind, Size, TyAbiInterface};

use crate::callconv::{ArgAbi, CastTarget, FnAbi, Uniform};

fn classify_aggregate_type<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    let size = arg.layout.size;

    match size.bits() {
        0 => return,
        1..=64 => {
            arg.cast_to(Reg { kind: RegKind::Integer, size });
        }
        65..=128 => {
            arg.cast_to(CastTarget::from(Uniform::new(Reg::i64(), Size::from_bytes(16))));
        }
        _ => {
            arg.make_indirect();
        }
    }
}

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if !ret.layout.is_sized() {
        // Not touching this...
        return;
    }

    if ret.layout.is_aggregate() || ret.layout.size.bits() > 64 {
        classify_aggregate_type(ret);
    } else {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if arg.layout.is_aggregate() || arg.layout.size.bits() > 64 {
        classify_aggregate_type(arg);
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
