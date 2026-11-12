use rustc_abi::{HasDataLayout, Size, TyAbiInterface};

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
    let size = ret.layout.size;
    if size.bytes() <= 8 {
        ret.cast_to(smallest_int_reg(size));
    } else {
        ret.make_indirect();
    }
}

/// The number of 32-bit registers (R0-R5) available for passing arguments.
const ARG_REGS: u32 = 6;

/// Account for an argument of the given `size` in the remaining argument registers.
///
/// Returns `true` if the argument is passed entirely in registers. Returns `false` if it does
/// not fit in the registers that are left; it is then passed on the stack, and no registers
/// remain for any subsequent arguments (even ones that would have fit in the skipped registers).
fn adjust_regs_left(size: Size, regs_left: &mut u32) -> bool {
    if *regs_left == 0 {
        return false;
    }
    if size.bytes() <= 4 {
        *regs_left -= 1;
        return true;
    }
    // Arguments larger than 32 bits are passed in an aligned register pair (R1:R0, R3:R2 or
    // R5:R4), so skip a register if the next free one is odd-numbered.
    if *regs_left >= 2 && !regs_left.is_multiple_of(2) {
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
        // The pointer to the indirectly passed value takes up a register.
        adjust_regs_left(Size::from_bytes(4), regs_left);
        return;
    }

    let size = arg.layout.size;

    if !arg.layout.is_aggregate() {
        if size.bytes() <= 8 {
            adjust_regs_left(size, regs_left);
        }
        arg.extend_integer_width_to(32);
        return;
    }

    if size.bytes() > 8 {
        arg.pass_by_stack_offset(None);
        return;
    }

    let in_regs = adjust_regs_left(size, regs_left);
    // In registers, aggregates have the alignment of the register(s) they are cast to. On the
    // stack, they just have their natural alignment.
    let align_bytes =
        if in_regs { if size.bytes() <= 4 { 4 } else { 8 } } else { arg.layout.align.bytes() };

    if size.bytes() <= align_bytes {
        arg.cast_to(smallest_int_reg(size));
    } else {
        arg.pass_by_stack_offset(None);
    }
}

/// The smallest integer register that can hold `size` (which is at most 8 bytes).
fn smallest_int_reg(size: Size) -> Reg {
    match size.bytes() {
        ..=1 => Reg::i8(),
        ..=2 => Reg::i16(),
        ..=4 => Reg::i32(),
        ..=8 => Reg::i64(),
        _ => unreachable!("aggregate of {} bytes does not fit in a register", size.bytes()),
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

    // Note that, matching clang, the hidden pointer for an indirect return value does not use up
    // any of the argument registers.
    let mut regs_left = ARG_REGS;
    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, &mut regs_left);
    }
}
