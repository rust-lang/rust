use rustc_abi::{BackendRepr, Float, HasDataLayout, Integer, Primitive, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi};

fn unwrap_trivial_aggregate<'a, Ty, C>(cx: &C, val: &mut ArgAbi<'a, Ty>) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if val.layout.is_aggregate() {
        if let Some(unit) = val.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()) {
            let size = val.layout.size;
            // This size check also catches over-aligned scalars as `size` will be rounded up to a
            // multiple of the alignment, and the default alignment of all scalar types on wasm
            // equals their size.
            if unit.size == size {
                val.cast_to(unit);
                return true;
            }
        }
    }
    false
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    ret.extend_integer_width_to(32);
    if ret.layout.is_aggregate() && !unwrap_trivial_aggregate(cx, ret) {
        ret.make_indirect();
    }

    // `long double`, `__int128_t` and `__uint128_t` use an indirect return
    if let BackendRepr::Scalar(scalar) = ret.layout.backend_repr {
        match scalar.primitive() {
            Primitive::Int(Integer::I128, _) | Primitive::Float(Float::F128) => {
                ret.make_indirect();
            }
            _ => {}
        }
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
    arg.extend_integer_width_to(32);
    if arg.layout.is_aggregate() && !unwrap_trivial_aggregate(cx, arg) {
        arg.make_indirect();
    }
}

/// The purpose of this ABI is to match the C ABI (aka clang) exactly.
pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}
