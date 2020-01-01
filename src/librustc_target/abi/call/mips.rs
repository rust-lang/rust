use crate::abi::call::{ArgAbi, FnAbi, Reg, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, Size, TyLayoutMethods};

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'_, Ty>, offset: &mut Size)
where
    Ty: TyLayoutMethods<'a, C>,
    C: LayoutOf<Ty = Ty> + HasDataLayout,
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
    } else {
        ret.make_indirect();
        *offset += cx.data_layout().pointer_size;
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'_, Ty>, offset: &mut Size)
where
    Ty: TyLayoutMethods<'a, C>,
    C: LayoutOf<Ty = Ty> + HasDataLayout,
{
    let dl = cx.data_layout();
    let size = arg.layout.size;
    let align = arg.layout.align.max(dl.i32_align).min(dl.i64_align).abi;

    if arg.layout.is_aggregate() {
        arg.cast_to(Uniform { unit: Reg::i32(), total: size });
        if !offset.is_aligned(align) {
            arg.pad_with(Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32);
    }

    *offset = offset.align_to(align) + size.align_to(align);
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'_, Ty>)
where
    Ty: TyLayoutMethods<'a, C>,
    C: LayoutOf<Ty = Ty> + HasDataLayout,
{
    let mut offset = Size::ZERO;
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, &mut offset);
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, &mut offset);
    }
}
