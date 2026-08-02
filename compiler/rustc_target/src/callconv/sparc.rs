use rustc_abi::{
    BackendRepr, Float, HasDataLayout, Integer, Numeric, Primitive, RegKind, TyAbiInterface,
};

use crate::callconv::{ArgAbi, ArgAttribute, CastTarget, FnAbi, Reg, Uniform};

/// C `long double` is IEEE binary128 on 32-bit SPARC, i.e. Rust's `f128`. It is passed and
/// returned indirectly.
fn is_long_double(repr: BackendRepr) -> bool {
    matches!(repr, BackendRepr::Scalar(scalar) if scalar.primitive() == Primitive::Float(Float::F128))
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, offset: &mut Size)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if let Some(component) = ret.layout.complex_number(cx) {
        let reg = Reg { kind: component.reg_kind(), size: component.size() };
        let mut cast = CastTarget::pair(reg, reg);

        match component {
            Numeric::Float(Float::F128) => {
                // long double _Complex is special in that it should be marked as inreg.
                // See Clang `SparcV8ABIInfo::classifyReturnType`.
                cast.attrs.set(ArgAttribute::InReg);
            }
            Numeric::Float(Float::F16)
            | Numeric::Int(Integer::I8 | Integer::I16 | Integer::I32, _) => {
                let size = ret.layout.size;
                cast = CastTarget::from(Reg { kind: RegKind::Integer, size });
            }
            _ => { /* default behavior */ }
        }

        ret.cast_to(cast);
    } else if is_long_double(ret.layout.backend_repr) || ret.layout.is_aggregate() {
        // `long double` is returned through an `sret` pointer, see Clang
        // `SparcV8ABIInfo::classifyReturnType`. The `sret` attribute is also what makes LLVM emit
        // the `unimp` marker after the call and return to `%o7+12` in the callee.
        ret.make_indirect();
        *offset += cx.data_layout().pointer_size();
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
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        *offset += dl.pointer_size();
        return;
    }
    if is_long_double(arg.layout.backend_repr) {
        // See Clang `SparcV8ABIInfo::classifyArgumentType`.
        arg.pass_by_stack_offset(None);
        *offset += dl.pointer_size();
        return;
    }
    let size = arg.layout.size;
    let align = arg.layout.align.abi.max(dl.i32_align).min(dl.i64_align);

    if let Some(component) = arg.layout.complex_number(cx) {
        if let Numeric::Int(Integer::I8 | Integer::I16 | Integer::I32, _) = component {
            arg.cast_to(Reg { kind: RegKind::Integer, size: 2 * component.size() });
        } else {
            arg.pass_by_stack_offset(None);
        }
    } else if arg.layout.is_aggregate() {
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
            if arg.layout.is_zst() {
                arg.make_indirect_from_ignore();
                offset += cx.data_layout().pointer_size();
            }
            continue;
        }
        classify_arg(cx, arg, &mut offset);
    }
}
