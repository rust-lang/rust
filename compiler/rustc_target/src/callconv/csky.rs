// Reference: CSKY ABI Manual
// https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1695027452256/T-HEAD_800_Series_ABI_Standards_Manual.pdf
//
// Reference: Clang CSKY lowering code
// https://github.com/llvm/llvm-project/blob/4a074f32a6914f2a8d7215d78758c24942dddc3d/clang/lib/CodeGen/Targets/CSKY.cpp#L76-L162

use rustc_abi::{CanonAbi, TyAbiInterface};

use crate::callconv::{ArgAbi, ComputeCallConv, Reg, Uniform};

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if !ret.layout.is_sized() {
        // Not touching this...
        return;
    }
    // For return type, aggregate which <= 2*XLen will be returned in registers.
    // Otherwise, aggregate will be returned indirectly.
    if ret.layout.is_aggregate() {
        let total = ret.layout.size;
        if total.bits() > 64 {
            ret.make_indirect();
        } else if total.bits() > 32 {
            ret.cast_to(Uniform::new(Reg::i32(), total));
        } else {
            ret.cast_to(Reg::i32());
        }
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
    // For argument type, the first 4*XLen parts of aggregate will be passed
    // in registers, and the rest will be passed in stack.
    // So we can coerce to integers directly and let backend handle it correctly.
    if arg.layout.is_aggregate() {
        let total = arg.layout.size;
        if total.bits() > 32 {
            arg.cast_to(Uniform::new(Reg::i32(), total));
        } else {
            arg.cast_to(Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) struct CSkyCallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for CSkyCallConv
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
