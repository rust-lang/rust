use rustc_abi::{BackendRepr, Float, HasDataLayout, Primitive, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi};

fn is_long_double(repr: BackendRepr) -> bool {
    matches!(repr, BackendRepr::Scalar(scalar) if scalar.primitive() == Primitive::Float(Float::F128))
}

fn classify_ret<'a, Ty>(ret: &mut ArgAbi<'a, Ty>) {
    if is_long_double(ret.layout.backend_repr) || ret.layout.is_aggregate() {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }

    if is_long_double(arg.layout.backend_repr) || arg.layout.is_aggregate() {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            if arg.layout.is_zst() {
                arg.make_indirect_from_ignore();
            }
            continue;
        }
        classify_arg(cx, arg);
    }
}
