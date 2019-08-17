use crate::abi::call::{FnType, ArgType, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

fn classify_ret_ty<'a, Ty, C>(cx: &C, ret: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if ret.layout.is_aggregate() {
        if let Some(unit) = ret.layout.homogeneous_aggregate(cx).unit() {
            let size = ret.layout.size;
            if unit.size == size {
                ret.cast_to(Uniform {
                    unit,
                    total: size
                });
                return;
            }
        }

        ret.make_indirect();
    }
}

fn classify_arg_ty<Ty>(arg: &mut ArgType<'_, Ty>) {
    if arg.layout.is_aggregate() {
        arg.make_indirect_byval();
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
        classify_arg_ty(arg);
    }
}
