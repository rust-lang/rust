use rustc_abi::{HasDataLayout, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg};

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
    let bits = ret.layout.size.bits();
    if bits <= 64 {
        ret.cast_to(smallest_int_reg(bits));
    } else {
        ret.make_indirect();
    }
}

const ARG_REGS: u32 = 6;

fn adjust_regs_left(bits: u64, regs_left: &mut u32) -> bool {
    if *regs_left == 0 {
        return false;
    }
    if bits <= 32 {
        *regs_left -= 1;
        return true;
    }
    if *regs_left >= 2 && *regs_left % 2 == 1 {
        *regs_left -= 1;
    }
    if *regs_left >= 2 {
        *regs_left -= 2;
        true
    } else {
        *regs_left = 0;
        false
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, regs_left: &mut u32)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        adjust_regs_left(32, regs_left);
        return;
    }

    let bits = arg.layout.size.bits();

    if !arg.layout.is_aggregate() {
        if bits <= 64 {
            adjust_regs_left(bits, regs_left);
        }
        arg.extend_integer_width_to(32);
        return;
    }

    if bits > 64 {
        arg.pass_by_stack_offset(None);
        return;
    }

    let in_regs = adjust_regs_left(bits, regs_left);
    let align_bits =
        if in_regs { if bits <= 32 { 32 } else { 64 } } else { arg.layout.align.abi.bits() };

    if bits <= align_bits {
        arg.cast_to(smallest_int_reg(bits));
    } else {
        arg.pass_by_stack_offset(None);
    }
}

fn smallest_int_reg(bits: u64) -> Reg {
    if bits <= 8 {
        Reg::i8()
    } else if bits <= 16 {
        Reg::i16()
    } else if bits <= 32 {
        Reg::i32()
    } else {
        Reg::i64()
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

    let mut regs_left = ARG_REGS;
    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, &mut regs_left);
    }
}
