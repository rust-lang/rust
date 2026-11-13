use rustc_abi::{CanonAbi, TyAbiInterface};

use crate::callconv::{ArgAbi, ComputeCallConv};

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() {
        ret.make_indirect();
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
    if arg.layout.is_aggregate() {
        arg.pass_by_stack_offset(None);
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) struct M64KCallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for M64KCallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn compute_ret_abi(&mut self, _cx: &C, _conv: CanonAbi, ret: &mut ArgAbi<'a, Ty>) {
        if !ret.is_ignore() {
            classify_ret(ret);
        }
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        _conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        if arg.is_ignore() {
            return;
        }

        classify_arg(cx, arg);
    }
}
