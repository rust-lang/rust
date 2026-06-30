// Reference: MSP430 Embedded Application Binary Interface
// https://www.ti.com/lit/an/slaa534a/slaa534a.pdf

use rustc_type_ir::{Interner, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi};

// 3.5 Structures or Unions Passed and Returned by Reference
//
// "Structures (including classes) and unions larger than 32 bits are passed and
// returned by reference. To pass a structure or union by reference, the caller
// places its address in the appropriate location: either in a register or on
// the stack, according to its position in the argument list. (..)"
fn classify_ret<I: Interner>(ret: &mut ArgAbi<I>) {
    if ret.layout.is_aggregate() && ret.layout.size.bits() > 32 {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(16);
    }
}

fn classify_arg<I: Interner, C>(cx: &C, arg: &mut ArgAbi<I>)
where
    I: TyAbiInterface<C>,
{
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if arg.layout.is_aggregate() && arg.layout.size.bits() > 32 {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(16);
    }
}

pub(crate) fn compute_abi_info<I: Interner, C>(cx: &C, fn_abi: &mut FnAbi<I>)
where
    I: TyAbiInterface<C>,
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
