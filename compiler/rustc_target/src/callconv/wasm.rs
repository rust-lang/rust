use rustc_abi::{
    BackendRepr, Float, HasDataLayout, Integer, Primitive, Reg, RegKind, TyAbiInterface,
    TyAndLayout,
};

use crate::callconv::{ArgAbi, FnAbi};

fn singleton_scalar<'a, Ty, C>(cx: &C, layout: TyAndLayout<'a, Ty>) -> Option<Reg>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // The base case: a single scalar is a singleton scalar.
    if !layout.is_aggregate() {
        let BackendRepr::Scalar(scalar) = layout.backend_repr else {
            return None;
        };
        let kind = match scalar.primitive() {
            Primitive::Int(..) | Primitive::Pointer(_) => RegKind::Integer,
            Primitive::Float(_) => RegKind::Float,
        };
        return Some(Reg { kind, size: layout.size });
    }

    let mut found = None;
    for i in 0..layout.fields.count() {
        let field = layout.field(cx, i);
        if field.is_zst() {
            continue;
        }
        if found.is_some() {
            // A second member, so not a singleton.
            return None;
        }
        found = Some(singleton_scalar(cx, field)?);
    }

    // Reject over-aligned types.
    found.filter(|scalar| scalar.size == layout.size)
}

/// Return whether the value should be passed as an aggregate (i.e. indirectly).
fn is_aggregate_for_abi<'a, Ty, C>(cx: &C, val: &mut ArgAbi<'a, Ty>) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !val.layout.is_aggregate() {
        return false;
    }

    let Some(scalar) = singleton_scalar(cx, val.layout) else {
        return true;
    };

    val.cast_to(scalar);
    false
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // `long double`, `__int128_t` and `__uint128_t` use an indirect return
    if let BackendRepr::Scalar(scalar) = ret.layout.backend_repr
        && matches!(
            scalar.primitive(),
            Primitive::Int(Integer::I128, _) | Primitive::Float(Float::F128)
        )
    {
        ret.make_indirect();
        return;
    }

    ret.extend_integer_width_to(32);
    if is_aggregate_for_abi(cx, ret) {
        ret.make_indirect();
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
    arg.extend_integer_width_to(32);
    if is_aggregate_for_abi(cx, arg) {
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
