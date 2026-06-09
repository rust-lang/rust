use rustc_abi::{HasDataLayout, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg, Uniform};

fn classify_ret<'a, Ty, C>(_cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !ret.layout.is_sized() {
        return;
    }

    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
        return;
    }

    // Per the Hexagon ABI:
    // - Aggregates up to 32 bits are returned in R0
    // - Aggregates 33-64 bits are returned in R1:R0
    // - Aggregates > 64 bits are returned indirectly via hidden first argument
    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 32 {
        ret.cast_to(Uniform::new(Reg::i32(), size));
    } else if bits <= 64 {
        ret.cast_to(Uniform::new(Reg::i64(), size));
    } else {
        ret.make_indirect();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }

    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(32);
        return;
    }

    // Per the Hexagon ABI:
    // - Aggregates up to 32 bits are passed in a single register
    // - Aggregates 33-64 bits are passed in a register pair
    // - Aggregates > 64 bits are passed on the stack
    let size = arg.layout.size;
    let bits = size.bits();
    if bits <= 32 {
        arg.cast_to(Uniform::new(Reg::i32(), size));
    } else if bits <= 64 {
        arg.cast_to(Uniform::new(Reg::i64(), size));
    } else {
        arg.pass_by_stack_offset(None);
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}
