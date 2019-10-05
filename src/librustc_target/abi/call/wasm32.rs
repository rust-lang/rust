use crate::abi::call::{FnType, ArgType, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

fn unwrap_trivial_aggregate<'a, Ty, C>(cx: &C, val: &mut ArgType<'a, Ty>) -> bool
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if val.layout.is_aggregate() {
        if let Some(unit) = val.layout.homogeneous_aggregate(cx).unit() {
            let size = val.layout.size;
            if unit.size == size {
                val.cast_to(Uniform {
                    unit,
                    total: size
                });
                return true;
            }
        }
    }
    false
}


fn classify_ret_ty<'a, Ty, C>(cx: &C, ret: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    ret.extend_integer_width_to(32);
    if ret.layout.is_aggregate() {
        if !unwrap_trivial_aggregate(cx, ret) {
            ret.make_indirect();
        }
    }
}

fn classify_arg_ty<'a, Ty, C>(cx: &C, arg: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    arg.extend_integer_width_to(32);
    if arg.layout.is_aggregate() {
        if !unwrap_trivial_aggregate(cx, arg) {
            arg.make_indirect_byval();
        }
    }
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fty: &mut FnType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg);
    }
}
