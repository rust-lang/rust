use crate::abi::call::{ArgType, FnType, Reg, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, Size, TyLayoutMethods};

fn classify_ret_ty<'a, Ty, C>(cx: &C, ret: &mut ArgType<'_, Ty>, offset: &mut Size)
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> + HasDataLayout
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
    } else {
        ret.make_indirect();
        *offset += cx.data_layout().pointer_size;
    }
}

fn classify_arg_ty<'a, Ty, C>(cx: &C, arg: &mut ArgType<'_, Ty>, offset: &mut Size)
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> + HasDataLayout
{
    let dl = cx.data_layout();
    let size = arg.layout.size;
    let align = arg.layout.align.max(dl.i32_align).min(dl.i64_align).abi;

    if arg.layout.is_aggregate() {
        arg.cast_to(Uniform {
            unit: Reg::i32(),
            total: size
        });
        if !offset.is_aligned(align) {
            arg.pad_with(Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32);
    }

    *offset = offset.align_to(align) + size.align_to(align);
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fty: &mut FnType<'_, Ty>)
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> + HasDataLayout
{
    let mut offset = Size::ZERO;
    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret, &mut offset);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg, &mut offset);
    }
}
