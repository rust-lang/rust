// FIXME: This needs an audit for correctness and completeness.

use rustc_abi::{
    BackendRepr, FieldsShape, Float, HasDataLayout, Primitive, Reg, Scalar, Size, TyAbiInterface,
    TyAndLayout,
};

use crate::callconv::{ArgAbi, ArgAttribute, CastTarget, FnAbi, Uniform};
use crate::spec::HasTargetSpec;

#[derive(Clone, Debug)]
struct Sdata {
    pub prefix: [Option<Reg>; 8],
    pub prefix_index: usize,
    pub last_offset: Size,
    pub has_float: bool,
    pub arg_attribute: ArgAttribute,
}

fn arg_scalar<C>(cx: &C, scalar: &Scalar, offset: Size, mut data: Sdata) -> Sdata
where
    C: HasDataLayout,
{
    let dl = cx.data_layout();

    if !matches!(scalar.primitive(), Primitive::Float(Float::F32 | Float::F64)) {
        return data;
    }

    data.has_float = true;

    if !data.last_offset.is_aligned(dl.f64_align.abi) && data.last_offset < offset {
        if data.prefix_index == data.prefix.len() {
            return data;
        }
        data.prefix[data.prefix_index] = Some(Reg::i32());
        data.prefix_index += 1;
        data.last_offset = data.last_offset + Reg::i32().size;
    }

    for _ in 0..((offset - data.last_offset).bits() / 64)
        .min((data.prefix.len() - data.prefix_index) as u64)
    {
        data.prefix[data.prefix_index] = Some(Reg::i64());
        data.prefix_index += 1;
        data.last_offset = data.last_offset + Reg::i64().size;
    }

    if data.last_offset < offset {
        if data.prefix_index == data.prefix.len() {
            return data;
        }
        data.prefix[data.prefix_index] = Some(Reg::i32());
        data.prefix_index += 1;
        data.last_offset = data.last_offset + Reg::i32().size;
    }

    if data.prefix_index == data.prefix.len() {
        return data;
    }

    if scalar.primitive() == Primitive::Float(Float::F32) {
        data.arg_attribute = ArgAttribute::InReg;
        data.prefix[data.prefix_index] = Some(Reg::f32());
        data.last_offset = offset + Reg::f32().size;
    } else {
        data.prefix[data.prefix_index] = Some(Reg::f64());
        data.last_offset = offset + Reg::f64().size;
    }
    data.prefix_index += 1;
    data
}

fn arg_scalar_pair<C>(
    cx: &C,
    scalar1: &Scalar,
    scalar2: &Scalar,
    mut offset: Size,
    mut data: Sdata,
) -> Sdata
where
    C: HasDataLayout,
{
    data = arg_scalar(cx, scalar1, offset, data);
    match (scalar1.primitive(), scalar2.primitive()) {
        (Primitive::Float(Float::F32), _) => offset += Reg::f32().size,
        (_, Primitive::Float(Float::F64)) => offset += Reg::f64().size,
        (Primitive::Int(i, _signed), _) => offset += i.size(),
        (Primitive::Pointer(_), _) => offset += Reg::i64().size,
        _ => {}
    }

    if (offset.bytes() % 4) != 0
        && matches!(scalar2.primitive(), Primitive::Float(Float::F32 | Float::F64))
    {
        offset += Size::from_bytes(4 - (offset.bytes() % 4));
    }
    data = arg_scalar(cx, scalar2, offset, data);
    data
}

fn parse_structure<'a, Ty, C>(
    cx: &C,
    layout: TyAndLayout<'a, Ty>,
    mut data: Sdata,
    mut offset: Size,
) -> Sdata
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if let FieldsShape::Union(_) = layout.fields {
        return data;
    }

    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            data = arg_scalar(cx, &scalar, offset, data);
        }
        BackendRepr::Memory { .. } => {
            for i in 0..layout.fields.count() {
                if offset < layout.fields.offset(i) {
                    offset = layout.fields.offset(i);
                }
                data = parse_structure(cx, layout.field(cx, i), data.clone(), offset);
            }
        }
        _ => {
            if let BackendRepr::ScalarPair(scalar1, scalar2) = &layout.backend_repr {
                data = arg_scalar_pair(cx, scalar1, scalar2, offset, data);
            }
        }
    }

    data
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, in_registers_max: Size)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        return;
    }

    let total = arg.layout.size;
    if total > in_registers_max {
        arg.make_indirect();
        return;
    }

    match arg.layout.fields {
        FieldsShape::Primitive => unreachable!(),
        FieldsShape::Array { .. } => {
            // Arrays are passed indirectly
            arg.make_indirect();
            return;
        }
        FieldsShape::Union(_) => {
            // Unions and are always treated as a series of 64-bit integer chunks
        }
        FieldsShape::Arbitrary { .. } => {
            // Structures with floating point numbers need special care.

            let mut data = parse_structure(
                cx,
                arg.layout,
                Sdata {
                    prefix: [None; 8],
                    prefix_index: 0,
                    last_offset: Size::ZERO,
                    has_float: false,
                    arg_attribute: ArgAttribute::default(),
                },
                Size::ZERO,
            );

            if data.has_float {
                // Structure { float, int, int } doesn't like to be handled like
                // { float, long int }. Other way around it doesn't mind.
                if data.last_offset < arg.layout.size
                    && (data.last_offset.bytes() % 8) != 0
                    && data.prefix_index < data.prefix.len()
                {
                    data.prefix[data.prefix_index] = Some(Reg::i32());
                    data.prefix_index += 1;
                    data.last_offset += Reg::i32().size;
                }

                let mut rest_size = arg.layout.size - data.last_offset;
                if (rest_size.bytes() % 8) != 0 && data.prefix_index < data.prefix.len() {
                    data.prefix[data.prefix_index] = Some(Reg::i32());
                    rest_size = rest_size - Reg::i32().size;
                }

                arg.cast_to(
                    CastTarget::prefixed(data.prefix, Uniform::new(Reg::i64(), rest_size))
                        .with_attrs(data.arg_attribute.into()),
                );
                return;
            }
        }
    }

    arg.cast_to(Uniform::new(Reg::i64(), total));
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        classify_arg(cx, &mut fn_abi.ret, Size::from_bytes(32));
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            // sparc64-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
            if cx.target_spec().os == "linux"
                && matches!(&*cx.target_spec().env, "gnu" | "musl" | "uclibc")
                && arg.layout.is_zst()
            {
                arg.make_indirect_from_ignore();
            }
            return;
        }
        classify_arg(cx, arg, Size::from_bytes(16));
    }
}
