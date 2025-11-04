use rustc_abi::{HasDataLayout, Size, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg, Uniform};

fn classify_ret<Ty, C>(cx: &C, ret: &mut ArgAbi<'_, Ty>, offset: &mut Size)
where
    C: HasDataLayout,
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
    } else {
        ret.make_indirect();
        *offset += cx.data_layout().pointer_size();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, offset: &mut Size)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        // FIXME: Update offset?
        // Not touching this...
        return;
    }
    let dl = cx.data_layout();
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        *offset += dl.pointer_size();
        return;
    }
    let size = arg.layout.size;
    let align = arg.layout.align.abi.max(dl.i32_align).min(dl.i64_align);

    if arg.layout.is_aggregate() {
        let pad_i32 = !offset.is_aligned(align);
        arg.cast_to_and_pad_i32(Uniform::new(Reg::i32(), size), pad_i32);
    } else {
        arg.extend_integer_width_to(32);
    }

    *offset = offset.align_to(align) + size.align_to(align);
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    let mut offset = Size::ZERO;
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, &mut offset);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, &mut offset);
    }
}
