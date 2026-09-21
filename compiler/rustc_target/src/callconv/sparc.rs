use rustc_abi::{BackendRepr, Float, HasDataLayout, Primitive, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi};

fn classify<'a, Ty>(arg: &mut ArgAbi<'a, Ty>) {
    if arg.layout.is_aggregate() {
        arg.make_indirect();
    } else if let BackendRepr::Scalar(scalar) = arg.layout.backend_repr
        && scalar.primitive() == Primitive::Float(Float::F128)
    {
        // Always pass f128 indirectly.
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(32);
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

    classify(arg);
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify(&mut fn_abi.ret);
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
