use arrayvec::ArrayVec;
use rustc_abi::{
    BackendRepr, CanonAbi, FieldsShape, Float, HasDataLayout, Integer, Primitive, Reg,
    TyAbiInterface, TyAndLayout, Variants,
};

use crate::callconv::{ArgAbi, CastTarget, FnAbi};

fn primitive_homogeneous_aggregate<'a, Ty, C>(cx: &C, layout: &TyAndLayout<'a, Ty>) -> Option<Reg>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    let unit = layout.homogeneous_aggregate(cx).ok()?.unit()?;
    // This size check also catches over-aligned scalars as `size` will be rounded up to a
    // multiple of the alignment, and the default alignment of all scalar types on wasm
    // equals their size.
    if unit.size == layout.size {
        return Some(unit);
    }
    None
}

/// Creates a `CastTarget`, if possible, for the aggregate `val`.
///
/// This will see if `val` is made up of up to `limit` primitives, and if so an
/// appropriate `CastTarget` is created. If `val` is not appropriate to cast, or
/// has too many primitives, then `None` is returned.
fn aggregate_cast_target<'a, Ty, C>(
    cx: &C,
    val: &ArgAbi<'a, Ty>,
    limit: usize,
) -> Option<CastTarget>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    assert!(limit > 0);
    if let Some(reg) = primitive_homogeneous_aggregate(cx, &val.layout) {
        return Some(reg.into());
    }
    if limit == 1 {
        return None;
    }

    match val.layout.variants {
        // Only structs are considered for flattening in wasm, so discard
        // everything else.
        Variants::Empty | Variants::Multiple { .. } => return None,

        // Fall through to go see further...
        Variants::Single { .. } => {}
    }

    match val.layout.fields {
        // Primitives and single-element unions are handled with
        // `homogeneous_aggregate` above. Arrays are always passed indirectly in
        // wasm.
        FieldsShape::Primitive | FieldsShape::Union(_) | FieldsShape::Array { .. } => return None,

        // Fall through to go see further...
        FieldsShape::Arbitrary { .. } => {}
    }

    if val.layout.fields.count() > limit {
        return None;
    }

    let mut prefix = ArrayVec::new();
    for i in 0..val.layout.fields.count() {
        let field = val.layout.field(cx, i);
        let primitive = primitive_homogeneous_aggregate(cx, &field)?;
        prefix.push(primitive)
    }
    let suffix = prefix.pop().unwrap();
    Some(CastTarget::prefixed(prefix, suffix.into()))
}

/// Aggregates are handled differently in wasm depending on where they are
/// (param or result) and what ABI used (currently `wasm-multivalue` tweaks
/// these limits a bit). This handles everything internally if `val` is an
/// aggregate.
///
/// Namely this function will cast `val` to an appropriate type if `val` is an
/// aggregate made up of up to `limit` primitive values. Otherwise `val` is
/// passed indirectly.
fn handle_aggregate<'a, Ty, C>(cx: &C, val: &mut ArgAbi<'a, Ty>, limit: usize)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !val.layout.is_aggregate() {
        return;
    }
    match aggregate_cast_target(cx, val, limit) {
        Some(target) => val.cast_to(target),
        None => val.make_indirect(),
    }
}

fn classify_ret<'a, Ty, C>(cx: &C, abi: CanonAbi, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    ret.extend_integer_width_to(32);

    // The "wasm-multivalue" ABI for results enables using multiple returns at
    // all through the use of returning an aggregate-of-primitives. In this
    // situation there's a much higher limit to be able to describe all desired
    // wasm destination type signatures. Other ABIs, however, only allow at most
    // one field in aggregates (e.g. newtype wrappers).
    let limit = match abi {
        CanonAbi::WasmMultivalue => 1000,
        _ => 1,
    };
    handle_aggregate(cx, ret, limit);

    // `long double`, `__int128_t` and `__uint128_t` use an indirect return
    // unless this is the `wasm-multivalue` ABI.
    if abi != CanonAbi::WasmMultivalue
        && let BackendRepr::Scalar(scalar) = ret.layout.backend_repr
    {
        match scalar.primitive() {
            Primitive::Int(Integer::I128, _) | Primitive::Float(Float::F128) => {
                ret.make_indirect();
            }
            _ => {}
        }
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, abi: CanonAbi, arg: &mut ArgAbi<'a, Ty>)
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

    // The "wasm-multivalue" ABI can flatten structs of up to 2 fields, but no
    // more. Other ABIs only flatten at most one field in aggregates, e.g.
    // newtype structs.
    let limit = match abi {
        CanonAbi::WasmMultivalue => 2,
        _ => 1,
    };
    handle_aggregate(cx, arg, limit);
}

/// The purpose of this ABI is to match the C ABI (aka clang) exactly.
pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, fn_abi.conv, &mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, fn_abi.conv, arg);
    }
}
