use rustc_abi::HasDataLayout;
use rustc_type_ir::{Interner, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi};

fn classify_ret<I: Interner, C>(_cx: &C, ret: &mut ArgAbi<I>)
where
    I: TyAbiInterface<C>,
    C: HasDataLayout,
{
    ret.extend_integer_width_to(32);
}

fn classify_arg<I: Interner, C>(cx: &C, arg: &mut ArgAbi<I>)
where
    I: TyAbiInterface<C>,
    C: HasDataLayout,
{
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    arg.extend_integer_width_to(32);
}

pub(crate) fn compute_abi_info<I: Interner, C>(cx: &C, fn_abi: &mut FnAbi<I>)
where
    I: TyAbiInterface<C>,
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
