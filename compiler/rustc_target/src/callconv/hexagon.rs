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

    let size = ret.layout.size;
    let bits = size.bits();

    // Aggregates larger than 64 bits are returned indirectly
    if bits > 64 {
        ret.make_indirect();
        return;
    }

    // Small aggregates are returned in registers
    // Cast to appropriate register type to ensure proper ABI
    let align = ret.layout.align.bytes();
    ret.cast_to(Uniform::consecutive(if align <= 4 { Reg::i32() } else { Reg::i64() }, size));
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

    let size = arg.layout.size;
    let bits = size.bits();

    // Aggregates larger than 64 bits are passed indirectly
    if bits > 64 {
        arg.make_indirect();
        return;
    }

    // Small aggregates are passed in registers
    // Cast to consecutive register-sized chunks to match the C ABI
    let align = arg.layout.align.bytes();
    arg.cast_to(Uniform::consecutive(if align <= 4 { Reg::i32() } else { Reg::i64() }, size));
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
