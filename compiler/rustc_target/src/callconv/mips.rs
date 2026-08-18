use rustc_abi::{Float, HasDataLayout, Integer, Numeric, Reg, RegKind, Size, TyAbiInterface};

use crate::callconv::{ArgAbi, CastTarget, FnAbi, Uniform};

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, offset: &mut Size)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    let dl = cx.data_layout();
    let size = ret.layout.size;

    if let Some(component) = ret.layout.complex_number(cx) {
        match component {
            Numeric::Float(Float::F128) => {
                // Same as an aggregate.
                ret.make_indirect();
                *offset += dl.pointer_size();
            }
            Numeric::Int(Integer::I8 | Integer::I16, _) => {
                // Pack Complex<{integer}> into a single register if that fits.
                ret.cast_to(Reg { kind: RegKind::Integer, size });
            }
            _ => {
                let reg = Reg { kind: component.reg_kind(), size: ret.layout.field(cx, 0).size };
                ret.cast_to(CastTarget::pair(reg, reg));
            }
        }
    } else if ret.layout.is_aggregate() {
        ret.make_indirect();
        *offset += dl.pointer_size();
    } else {
        ret.extend_integer_width_to(32);
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
    let align = arg.layout.align.abi.max(dl.i32_align).min(dl.i64_align);

    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        *offset = offset.align_to(align) + dl.pointer_size().align_to(align);
        return;
    }

    let size = arg.layout.size;
    if arg.layout.is_aggregate() {
        let pad_i32 = u8::from(!offset.is_aligned(align));
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
