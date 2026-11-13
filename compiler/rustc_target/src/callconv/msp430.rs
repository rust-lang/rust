// Reference: MSP430 Embedded Application Binary Interface
// https://www.ti.com/lit/an/slaa534a/slaa534a.pdf

use rustc_abi::{CanonAbi, TyAbiInterface};

use crate::callconv::{ArgAbi, ComputeCallConv};

// 3.5 Structures or Unions Passed and Returned by Reference
//
// "Structures (including classes) and unions larger than 32 bits are passed and
// returned by reference. To pass a structure or union by reference, the caller
// places its address in the appropriate location: either in a register or on
// the stack, according to its position in the argument list. (..)"
fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() && ret.layout.size.bits() > 32 {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(16);
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
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

pub(crate) struct MSP430CallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for MSP430CallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn passes_zst(&self) -> bool {
        true
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
        classify_arg(cx, arg);
    }
}
