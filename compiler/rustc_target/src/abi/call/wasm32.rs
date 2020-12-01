use crate::abi::call::{ArgAbi, FnAbi, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, TyAndLayout, TyAndLayoutMethods};

fn unwrap_trivial_aggregate<'a, Ty, C>(cx: &C, val: &mut ArgAbi<'a, Ty>) -> bool
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if val.layout.is_aggregate() {
        if let Some(unit) = val.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()) {
            let size = val.layout.size;
            if unit.size == size {
                val.cast_to(Uniform { unit, total: size });
                return true;
            }
        }
    }
    false
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    ret.extend_integer_width_to(32);
    if ret.layout.is_aggregate() && !unwrap_trivial_aggregate(cx, ret) {
        ret.make_indirect();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    arg.extend_integer_width_to(32);
    if arg.layout.is_aggregate() && !unwrap_trivial_aggregate(cx, arg) {
        arg.make_indirect_byval();
    }
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}
